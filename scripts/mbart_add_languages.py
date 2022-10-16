"""
Adds special language tags (e.g., de_COMPLEX, de_SIMPLE) to the mBART vocabulary.

Based on Rios et al. (2022). See https://github.com/a-rios/longmbart/blob/e250cfc100401e263409bb147ed2f0ae9dfdc1c4/scripts/convert_mbart_to_longformerencoderdecoder.py
"""

import argparse

import torch
from tokenizers import AddedToken
from transformers import MBartForConditionalGeneration, MBartTokenizer


def main(args):
    tokenizer = MBartTokenizer.from_pretrained(args.model_name_or_path)
    model = MBartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    print("loaded tokenizer with len ", len(tokenizer.sp_model))

    if args.add_language_tags is not None:
        embed_weight = model.model.shared.weight  # (vocab, dim)
        print(embed_weight.shape)
        ## need to reduce final_logits_bias too
        final_logits_bias = model.final_logits_bias.transpose(0, 1)  # (1, vocab_size)
        # print("new model, logits bias ", final_logits_bias)
        # print("new model, logits bias non zero", final_logits_bias.nonzero())

        print(tokenizer._additional_special_tokens)
        print("tokenizer orig len ", tokenizer.vocab_size)
        current_tags = tokenizer.special_tokens_map["additional_special_tokens"]
        special_tokens = {"additional_special_tokens": current_tags + args.add_language_tags}
        tokenizer.add_special_tokens(special_tokens)
        print("tokenizer len ", tokenizer.vocab_size)

        for (new_tag, init_tag) in zip(args.add_language_tags, args.initialize_tags):
            init_tag_id = tokenizer.lang_code_to_id[init_tag]
            print("init_tag_id ", init_tag_id)
            init_embed = model.model.shared.weight[init_tag_id].unsqueeze(0)
            embed_weight = torch.cat((embed_weight, init_embed), dim=0)  # pylint: disable=E1101
            init_bias = final_logits_bias[init_tag_id].unsqueeze(dim=0)
            final_logits_bias = torch.cat((final_logits_bias, init_bias), dim=0)  # pylint: disable=E1101
            print("added ", new_tag)
            print("tag embedding shape ", init_embed.shape)
            print("embedding matrix shape ", embed_weight.shape)

        model.final_logits_bias.data = final_logits_bias.transpose(0, 1)
        model.model.shared.weight.data = embed_weight
        model.config.vocab_size = embed_weight.shape[0]

        print("saving tokenizer with new tags")
        tokenizer.save_pretrained(args.save_model_to)
        print("saving model with new tags")
        model.save_pretrained(args.save_model_to)

    print("special tokens map ", tokenizer.special_tokens_map)
    print("id-to-lang-code ", tokenizer.id_to_lang_code)
    print("lang-code-to-id", tokenizer.lang_code_to_id)

    ## check embeddings
    if args.add_language_tags is not None and args.initialize_tags is not None:
        for new_tag, init_tag in zip(args.add_language_tags, args.initialize_tags):
            print(
                "original language embedding for {}: {}".format(
                    init_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(init_tag)]
                )
            )
            print(
                "initialized {} with embedding: {}".format(
                    new_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(new_tag)]
                )
            )


def arg_parser():
    parser = argparse.ArgumentParser(description="Add special language tokens to mBART.")
    parser.add_argument(
        "--model_name_or_path", type=str, help="The name or path of the base model you want to convert"
    )
    parser.add_argument("--save_model_to", type=str, required=True, help="The path to save the converted model")
    parser.add_argument(
        "--add_language_tags",
        type=str,
        nargs="+",
        help="List of additional language tags (will replace tags given with --replace_tags and initialize with embeddings given with --initialize_tags).",
    )
    parser.add_argument(
        "--initialize_tags", type=str, nargs="+", help="Initialize new language tags with embeddings of these tags."
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Levels of verbosity affect what is tested/shown after converting model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())
