import Foundation
@testable import Cotabby

/// A twelve-token byte-level vocabulary shaped like the real Qwen3/Llama 3 `tokenizer.json` files,
/// shared by the Neural Engine vocabulary and sampler tests so both read the same ids.
///
/// Byte-level BPE spells a space as "Ġ" (U+0120) and a newline as "Ċ" (U+010A); "Ã©" is the two
/// UTF-8 bytes of "é".
enum NeuralEngineTestVocabulary {
    static let hello: Int32 = 0           // "hello"
    static let world: Int32 = 1           // " world"
    static let the: Int32 = 2             // " the"
    static let newline: Int32 = 3         // "\n"
    static let ing: Int32 = 4             // "ing"
    static let int: Int32 = 5             // " int"
    static let intelligence: Int32 = 6    // " intelligence"
    static let eAcute: Int32 = 7          // "é"
    static let markerInVocab: Int32 = 8   // "<|im_start|>" shipped as an ordinary token
    static let endOfText: Int32 = 9       // special, ends generation
    static let beginOfText: Int32 = 10    // special BOS
    static let unassigned: Int32 = 11     // a logit the tokenizer never produces
    static let size = 12

    static func json(prependsBOS: Bool = true, decoder: String = "ByteLevel") -> Data {
        var root: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": [
                    "hello": 0, "Ġworld": 1, "Ġthe": 2, "Ċ": 3, "ing": 4,
                    "Ġint": 5, "Ġintelligence": 6, "Ã©": 7, "<|im_start|>": 8
                ]
            ],
            "added_tokens": [
                ["id": 9, "content": "<|endoftext|>", "special": true],
                ["id": 10, "content": "<|begin_of_text|>", "special": true]
            ],
            "decoder": ["type": decoder]
        ]
        if prependsBOS {
            root["post_processor"] = [
                "type": "Sequence",
                "processors": [
                    ["type": "ByteLevel"],
                    [
                        "type": "TemplateProcessing",
                        "single": [
                            ["SpecialToken": ["id": "<|begin_of_text|>", "type_id": 0]],
                            ["Sequence": ["id": "A", "type_id": 0]]
                        ],
                        "special_tokens": [
                            "<|begin_of_text|>": ["id": "<|begin_of_text|>", "ids": [10], "tokens": ["<|begin_of_text|>"]]
                        ]
                    ]
                ]
            ]
        }
        return try! JSONSerialization.data(withJSONObject: root)
    }

    static func make(prependsBOS: Bool = true) -> NeuralEngineVocabulary {
        try! NeuralEngineVocabulary(tokenizerJSON: json(prependsBOS: prependsBOS), vocabularySize: size)
    }
}
