import json

def extract_mixset(
    input_path,
    human_out_path,
    ai_out_path,
    ai_key
):
    human_out = open(human_out_path, "w", encoding="utf-8")
    ai_out = open(ai_out_path, "w", encoding="utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for sample in data:
            sid = sample.get("id", None)
            
            h_text = sample.get("HWT_sentence")
            a_text = sample.get(ai_key)

            if h_text and a_text:
                h_record = {"id": sid, "text": h_text}
                a_record = {"id": sid, "text": a_text}
                
                human_out.write(json.dumps(h_record, ensure_ascii=False) + "\n")
                ai_out.write(json.dumps(a_record, ensure_ascii=False) + "\n")
    human_out.close()
    ai_out.close()


#extract_mixset("data/pair1_free/mixset_rewrite/2llama_rewrite.json", 
#                         "data/pair1_free/mixset_rewrite/human.jsonl", 
#                         "data/pair1_free/mixset_rewrite/ai_llama.jsonl", 
#                         "llama_rewrite_output")


#extract_mixset("data/pair1_free/mixset_rewrite/gpt4_rewrite.json", 
#                         "data/pair1_free/mixset_rewrite/human.jsonl", 
#                         "data/pair1_free/mixset_rewrite/ai_gpt4.jsonl", 
#                        "gpt4_rewrite_output")

# --- Pair 3: Imitation (Medium & Hard) ---
#extract_mixset("data/pair3_imitation/mixset_polish_sentence/2llama_polish_sentence.json", 
#                         "data/pair3_imitation/mixset_polish_sentence/human.jsonl", 
#                        "data/pair3_imitation/mixset_polish_sentence/ai_llama.jsonl", 
#                        "llama_polish_sentence_output")


#extract_mixset("data/pair3_imitation/mixset_polish_token/2llama_polish_token.json", 
#                        "data/pair3_imitation/mixset_polish_token/human.jsonl", 
#                        "data/pair3_imitation/mixset_polish_token/ai_llama.jsonl", 
#                       "llama_polish_token_output")

#extract_mixset("data/pair3_imitation/mixset_polish_sentence/gpt4_polish_sentence.json", 
#                         "data/pair3_imitation/mixset_polish_sentence/human.jsonl", 
#                         "data/pair3_imitation/mixset_polish_sentence/ai_gpt4.jsonl", 
#                         "gpt4_polish_sentence_output")


extract_mixset("data/pair3_imitation/mixset_polish_token/gpt4_polish_token.json", 
                         "data/pair3_imitation/mixset_polish_token/human.jsonl", 
                         "data/pair3_imitation/mixset_polish_token/ai_gpt4.jsonl", 
                         "gpt4_polish_token_output")