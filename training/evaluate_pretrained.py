"""Evaluate GPT-2/BERT on Turkish sentences for comparison"""
import sys, os, torch, json, argparse
from typing import List, Dict

ROOT = "/Users/serkankose/Documents/quann-3d"
sys.path.insert(0, ROOT)
from cloudlm import BPETokenizer

def load_corpus(path, max_lines=None, min_len=20, max_len=300):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            t = line.strip()
            if min_len <= len(t) <= max_len:
                texts.append(t)
    return texts

def evaluate_gpt2_turkish(texts: List[str], tok: BPETokenizer, max_samples=1000):
    """Evaluate Turkish GPT-2 on corpus"""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("Loading GPT-2 (Turkish)...")
        
        # Turkish GPT-2 model
        model_name = "dbmdz/gpt2-large-turkish"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for i, text in enumerate(texts[:max_samples]):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()
                n_samples += 1
                
                if (i + 1) % 100 == 0:
                    print(f"  GPT-2: {i+1}/{max_samples} samples")
        
        avg_loss = total_loss / max(n_samples, 1)
        ppl = math.exp(min(avg_loss, 20))
        
        return {
            "model": "GPT-2-Turkish",
            "ppl": ppl,
            "avg_loss": avg_loss,
            "samples": n_samples
        }
        
    except ImportError:
        print("  ⚠️  transformers not installed, skipping GPT-2")
        return None
    except Exception as e:
        print(f"  ⚠️  GPT-2 evaluation failed: {e}")
        return None

def evaluate_bert_turkish(texts: List[str], tok: BPETokenizer, max_samples=1000):
    """Evaluate Turkish BERT (masked LM perplexity)"""
    try:
        from transformers import BertForMaskedLM, BertTokenizer
        print("Loading BERT (Turkish)...")
        
        model_name = "dbmdz/bert-large-turkish"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        # BERT perplexity calculation (masked LM)
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for i, text in enumerate(texts[:max_samples]):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Create masked inputs
                masked_inputs = inputs["input_ids"].clone()
                labels = masked_inputs.clone()
                
                outputs = model(masked_inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                n_samples += 1
                
                if (i + 1) % 100 == 0:
                    print(f"  BERT: {i+1}/{max_samples} samples")
        
        avg_loss = total_loss / max(n_samples, 1)
        ppl = math.exp(min(avg_loss, 20))
        
        return {
            "model": "BERT-Turkish",
            "ppl": ppl,
            "avg_loss": avg_loss,
            "samples": n_samples
        }
        
    except ImportError:
        print("  ⚠️  transformers not installed, skipping BERT")
        return None
    except Exception as e:
        print(f"  ⚠️  BERT evaluation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--save-dir", default=os.path.join(ROOT, "checkpoints", "wiki_checkpoints_r2_cloud_attn_1500k_d64_n20"))
    parser.add_argument("--eval-gpt2", action="store_true")
    parser.add_argument("--eval-bert", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  PRETRAINED MODEL EVALUATION (GPT-2/BERT)")
    print("=" * 60)
    
    # Load corpus
    texts = load_corpus(args.corpus, args.max_samples)
    print(f"Corpus: {len(texts)} samples")
    
    # Load our tokenizer for consistency
    tok_path = os.path.join(args.save_dir, "wiki_bpe.json")
    if os.path.exists(tok_path):
        tok = BPETokenizer.load(tok_path)
        print(f"Loaded BPE: {tok.num_tokens} tokens")
    else:
        print("⚠️  No BPE found, using default")
        tok = None
    
    results = {}
    
    if args.eval_gpt2:
        print("\n📊 Evaluating GPT-2 Turkish...")
        gpt2_result = evaluate_gpt2_turkish(texts, tok, args.max_samples)
        if gpt2_result:
            results["gpt2"] = gpt2_result
            print(f"  ✅ GPT-2 PPL: {gpt2_result['ppl']:.2f}")
    
    if args.eval_bert:
        print("\n📊 Evaluating BERT Turkish...")
        bert_result = evaluate_bert_turkish(texts, tok, args.max_samples)
        if bert_result:
            results["bert"] = bert_result
            print(f"  ✅ BERT PPL: {bert_result['ppl']:.2f}")
    
    # Save results
    output_path = os.path.join(args.save_dir, "pretrained_comparison.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for model, result in results.items():
        print(f"  {model}: PPL = {result['ppl']:.2f}")

if __name__ == "__main__":
    import math
    main()
