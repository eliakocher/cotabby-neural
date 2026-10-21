import Foundation

/// Per-token facts the Neural Engine runtime needs in order to sample exactly like the llama path,
/// read once from a Core AI bundle's Hugging Face `tokenizer.json`.
///
/// llama.cpp answers these questions natively from GGUF metadata (`buildTokenMasks` in
/// CotabbyInference). Core AI returns raw logits and nothing else, so this type rebuilds the same
/// classification in Swift:
/// - every token's exact bytes, because token healing compares bytes and ghost text is assembled
///   from them without lossy decoding;
/// - tokens that may never be sampled as text: special/control tokens, ids the tokenizer never
///   assigns, the BOS token, and chat/FIM scaffolding markers;
/// - tokens containing a line break (masked in single-line fields) and tokens starting with
///   whitespace (masked for the first token when the caret sits inside a word);
/// - tokens that end generation (end-of-text and end-of-turn markers);
/// - the special token the tokenizer's post-processor puts in front of every prompt (Llama 3's
///   `<|begin_of_text|>`; Qwen3 has none). llama.cpp adds it from GGUF metadata, so the Neural
///   Engine path must add it too or the model sees a prompt shape it was never trained on.
///
/// Only byte-level BPE vocabularies (GPT-2 style, used by Qwen and Llama 3) are supported; their
/// token strings encode raw bytes through a fixed byte-to-character table that is inverted here.
/// Built off the main actor at model load (128k-152k entries) and immutable afterwards.
nonisolated struct NeuralEngineVocabulary: Sendable {
    enum LoadError: LocalizedError, Equatable {
        case malformed(String)
        case unsupportedTokenizer(String)

        var errorDescription: String? {
            switch self {
            case .malformed(let detail):
                return "The model's tokenizer.json is malformed: \(detail)"
            case .unsupportedTokenizer(let detail):
                return "The model's tokenizer is not supported by the Neural Engine runtime: \(detail)"
            }
        }
    }

    /// Number of logits the model produces; may exceed the tokenizer's assigned ids (padding).
    let size: Int
    /// Exact bytes per token id. Empty for special and unassigned ids, which render no text.
    let pieces: [[UInt8]]
    /// Never sampled as visible text. End-of-generation tokens stay sampleable so a natural stop
    /// is still recognized, exactly like the native mask.
    let isNonprintable: [Bool]
    let containsLineBreak: [Bool]
    let startsNewWord: [Bool]
    /// The same three classifications as id lists, so the sampler can stamp a mask into a logits
    /// row in one pass over the affected ids instead of testing all 150k ids per step.
    let nonprintableIDs: [Int32]
    let lineBreakIDs: [Int32]
    let newWordIDs: [Int32]
    let endOfGeneration: Set<Int32>
    /// Prepended to every encoded prompt when the tokenizer's post-processor template asks for it.
    let prependedSpecialToken: Int32?
    let healing: TokenHealingVocabulary

    /// Special tokens that mean "the text is over" across the supported model families.
    static let endOfGenerationMarkers: Set<String> = [
        "<|endoftext|>", "<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "<|eom_id|>",
        "<end_of_turn>", "<|end|>", "<eos>", "</s>"
    ]

    init(tokenizerJSON data: Data, vocabularySize: Int, additionalEndOfGeneration: Set<Int32> = []) throws {
        guard vocabularySize > 0 else { throw LoadError.malformed("vocabulary size is zero") }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LoadError.malformed("top level is not an object")
        }
        guard Self.usesByteLevelDecoder(root["decoder"]) else {
            throw LoadError.unsupportedTokenizer("only byte-level BPE vocabularies are supported")
        }
        guard let model = root["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Any] else {
            throw LoadError.malformed("missing model.vocab")
        }

        var pieces = [[UInt8]](repeating: [], count: vocabularySize)
        for (tokenString, value) in vocab {
            guard let id = (value as? NSNumber)?.intValue, id >= 0, id < vocabularySize else { continue }
            pieces[id] = Self.byteLevelBytes(tokenString)
        }

        // Added tokens are stored as literal text rather than byte-level characters. Special ones
        // (BOS, EOS, chat markers) render nothing as text, like llama.cpp's control tokens;
        // ordinary added tokens render their content.
        var isSpecial = [Bool](repeating: false, count: vocabularySize)
        var specialIDsByContent: [String: Int32] = [:]
        var endOfGeneration = additionalEndOfGeneration
        for case let added as [String: Any] in root["added_tokens"] as? [Any] ?? [] {
            guard let id = (added["id"] as? NSNumber)?.intValue, id >= 0, id < vocabularySize,
                  let content = added["content"] as? String else { continue }
            if (added["special"] as? Bool) ?? false {
                isSpecial[id] = true
                pieces[id] = []
                specialIDsByContent[content] = Int32(id)
                if Self.endOfGenerationMarkers.contains(content) { endOfGeneration.insert(Int32(id)) }
            } else {
                pieces[id] = Array(content.utf8)
            }
        }

        let prepended = Self.prependedSpecialToken(root["post_processor"], specialIDsByContent: specialIDsByContent)

        var isNonprintable = [Bool](repeating: false, count: vocabularySize)
        var containsLineBreak = [Bool](repeating: false, count: vocabularySize)
        var startsNewWord = [Bool](repeating: false, count: vocabularySize)
        var healingPieces: [(token: Int32, bytes: [UInt8])] = []
        for id in 0 ..< vocabularySize {
            let token = Int32(id)
            let piece = pieces[id]
            let isEnd = endOfGeneration.contains(token)
            if !isEnd {
                isNonprintable[id] = isSpecial[id] || piece.isEmpty || token == prepended
                    || Self.isScaffoldingMarker(piece)
            }
            if let first = piece.first, first == 32 || first == 9 || first == 10 || first == 13 {
                startsNewWord[id] = true
            }
            if !isEnd, piece.contains(10) || piece.contains(13) {
                containsLineBreak[id] = true
            }
            if !isNonprintable[id], !isEnd, !piece.isEmpty {
                healingPieces.append((token, piece))
            }
        }

        size = vocabularySize
        self.pieces = pieces
        self.isNonprintable = isNonprintable
        self.containsLineBreak = containsLineBreak
        self.startsNewWord = startsNewWord
        nonprintableIDs = isNonprintable.indices.filter { isNonprintable[$0] }.map(Int32.init)
        lineBreakIDs = containsLineBreak.indices.filter { containsLineBreak[$0] }.map(Int32.init)
        newWordIDs = startsNewWord.indices.filter { startsNewWord[$0] }.map(Int32.init)
        self.endOfGeneration = endOfGeneration
        prependedSpecialToken = prepended
        healing = TokenHealingVocabulary(pieces: healingPieces)
    }

    /// Bytes for `token`, or none for ids outside the vocabulary.
    func piece(_ token: Int32) -> [UInt8] {
        guard token >= 0, Int(token) < size else { return [] }
        return pieces[Int(token)]
    }

    // MARK: - tokenizer.json parsing

    private static func usesByteLevelDecoder(_ decoder: Any?) -> Bool {
        guard let decoder = decoder as? [String: Any], let type = decoder["type"] as? String else {
            return false
        }
        if type == "ByteLevel" { return true }
        if type == "Sequence", let steps = decoder["decoders"] as? [Any] {
            return steps.contains { usesByteLevelDecoder($0) }
        }
        return false
    }

    /// The special token the single-sequence template puts first, e.g. Llama 3's
    /// `[SpecialToken <|begin_of_text|>, Sequence A]`. The template may sit inside a `Sequence`
    /// post-processor next to a `ByteLevel` step.
    private static func prependedSpecialToken(_ postProcessor: Any?, specialIDsByContent: [String: Int32]) -> Int32? {
        guard let processor = postProcessor as? [String: Any], let type = processor["type"] as? String else {
            return nil
        }
        if type == "Sequence" {
            for step in processor["processors"] as? [Any] ?? [] {
                if let token = prependedSpecialToken(step, specialIDsByContent: specialIDsByContent) { return token }
            }
            return nil
        }
        guard type == "TemplateProcessing",
              let single = processor["single"] as? [Any],
              let first = single.first as? [String: Any],
              let special = first["SpecialToken"] as? [String: Any],
              let name = special["id"] as? String else {
            return nil
        }
        if let table = processor["special_tokens"] as? [String: Any],
           let entry = table[name] as? [String: Any],
           let id = (entry["ids"] as? [Any])?.first.flatMap({ ($0 as? NSNumber)?.int32Value }) {
            return id
        }
        return specialIDsByContent[name]
    }

    // MARK: - Byte-level BPE

    /// Inverse of GPT-2's `bytes_to_unicode`: printable Latin-1 bytes stand for themselves and the
    /// 68 others (controls, space, 0x7F-0xA0, soft hyphen) map to U+0100 onwards, in byte order.
    private static let byteForScalar: [UInt32: UInt8] = {
        var bytes = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var scalars = bytes.map(UInt32.init)
        var next: UInt32 = 256
        for byte in 0 ..< 256 where !bytes.contains(byte) {
            bytes.append(byte)
            scalars.append(next)
            next += 1
        }
        return Dictionary(uniqueKeysWithValues: zip(scalars, bytes.map { UInt8($0) }))
    }()

    static func byteLevelBytes(_ tokenString: String) -> [UInt8] {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(tokenString.unicodeScalars.count)
        for scalar in tokenString.unicodeScalars {
            if let byte = byteForScalar[scalar.value] {
                bytes.append(byte)
            } else {
                // Not a byte-level character: keep the scalar's own UTF-8 rather than dropping it.
                bytes.append(contentsOf: Array(String(scalar).utf8))
            }
        }
        return bytes
    }

    // MARK: - Scaffolding markers

    /// Same list as the native engine's `isScaffoldingMarkerPiece`: single-token chat, instruct
    /// and FIM markers that must never surface in autocomplete even when a vocabulary ships them
    /// as ordinary text tokens. Exact matches, plus prefixes for the FIM/repo families.
    private static let exactMarkers: Set<[UInt8]> = Set([
        "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<|system|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end|>", "<|endoftext|>",
        "<start_of_turn>", "<end_of_turn>", "[INST]", "[/INST]"
    ].map { Array($0.utf8) })
    private static let prefixMarkers: [[UInt8]] = ["<|fim_", "<fim_", "<|file_sep", "<|repo_name"].map { Array($0.utf8) }

    static func isScaffoldingMarker(_ piece: [UInt8]) -> Bool {
        guard piece.first == UInt8(ascii: "<") || piece.first == UInt8(ascii: "[") else { return false }
        return exactMarkers.contains(piece) || prefixMarkers.contains { piece.starts(with: $0) }
    }
}
