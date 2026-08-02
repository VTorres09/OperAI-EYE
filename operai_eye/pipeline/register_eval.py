"""Register an evaluation run in the metadata system.

Usage:
    python register_eval.py --model-id moondream2_v1 --model-name "Moondream2 2B" --prompt prompts/or_phase_simple.txt
"""

import argparse

from operai_eye.web.eval_data import register_model


def main():
    parser = argparse.ArgumentParser(description="Register an evaluation run")
    parser.add_argument("--model-id", required=True, help="Unique model ID")
    parser.add_argument("--model-name", required=True, help="Human-readable model name")
    parser.add_argument("--prompt", required=True, help="Prompt file used")
    parser.add_argument("--description", default="", help="Description of the run")

    args = parser.parse_args()

    result = register_model(
        model_id=args.model_id,
        model_name=args.model_name,
        prompt_file=args.prompt,
        description=args.description,
    )

    print(f"Registered model: {args.model_id}")
    print(f"  Name: {result['model_name']}")
    print(f"  Prompt: {result['prompt_file']}")
    print(f"  Results file: {result['results_file']}")


if __name__ == "__main__":
    main()
