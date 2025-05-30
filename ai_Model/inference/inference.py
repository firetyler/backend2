import math
import torch


def beam_search_generate(model, tokenizer, prompt, max_length=50, beam_width=3, eos_token_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Tokenisera prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    sequences = [(input_ids, 0.0)]  # (sequence_tensor, score)

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            with torch.no_grad():
                outputs = model(seq)
                logits = outputs[:, -1, :]  # sista token
                probs = torch.softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)

                for i in range(beam_width):
                    token_id = topk_indices[0][i].item()
                    token_prob = topk_probs[0][i].item()

                    new_token = torch.tensor([[token_id]], device=device)
                    new_seq = torch.cat([seq, new_token], dim=1)
                    new_score = score - math.log(token_prob + 1e-8)

                    all_candidates.append((new_seq, new_score))

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

        if eos_token_id is not None:
            if all(seq[0][0, -1].item() == eos_token_id for seq in sequences):
                break

    best_seq = sequences[0][0]
    return tokenizer.decode(best_seq.squeeze().tolist(), skip_special_tokens=True)

