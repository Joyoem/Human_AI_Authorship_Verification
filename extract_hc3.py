import json

def extract_hc3(
    input_path,
    human_out_path,
    ai_out_path
):
    human_out = open(human_out_path, "w", encoding="utf-8")
    ai_out = open(ai_out_path, "w", encoding="utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sample = json.loads(line)

            text = sample["text"]
            label = sample["label"]  # 0 = human, 1 = AI

            record = {
                "id": idx,
                "text": text
            }

            if label == 0:
                human_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                ai_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    human_out.close()
    ai_out.close()



if __name__ == "__main__":
    extract_hc3(
    input_path="data/pair3_imitation/hc3_si/test_hc3_si.jsonl", 
    human_out_path="data/pair2_semantic_preserving/hc3_si/human.jsonl",
    ai_out_path="data/pair2_semantic_preserving/hc3_si/ai.jsonl"
    )

    #extract_hc3(
    #input_path="data/pair1_free/hc3_qa/test_hc3_QA.jsonl", 
    #human_out_path="data/pair1_free/hc3_qa/human.jsonl",
    #ai_out_path="data/pair1_free/hc3_qa/ai.jsonl"
    #)