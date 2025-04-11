#!/usr/bin/env python
import json
import os
import argparse
from together import Together

def load_test_conversations(test_file):
    """
    Load test conversations from a JSON file.
    
    Args:
        test_file (str): Path to the test file
        
    Returns:
        list: List of test conversations
    """
    with open(test_file, 'r') as f:
        conversations = json.load(f)
    
    # Select a subset of conversations for testing
    # You can modify this to include all or specific conversations
    return conversations[:5]  # Use the first 5 conversations for testing

def evaluate_model(model_name, test_conversations, api_key=None):
    """
    Evaluate the model on test conversations.
    
    Args:
        model_name (str): Name of the model to evaluate
        test_conversations (list): List of test conversations
        api_key (str): Together API key
        
    Returns:
        dict: Evaluation results
    """
    client = Together(api_key=api_key or os.environ.get("TOGETHER_API_KEY"))
    results = []
    
    for i, conversation in enumerate(test_conversations):
        print(f"\nTesting conversation {i+1}/{len(test_conversations)}")
        
        # Initialize with system message
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
        
        # Process each turn in the conversation
        for j in range(0, len(conversation), 2):
            if j + 1 >= len(conversation):
                break
                
            user_msg = conversation[j]
            expected_assistant_msg = conversation[j + 1]
            
            # Add user message
            messages.append({"role": "user", "content": user_msg["content"]})
            
            print(f"  Turn {j//2 + 1}:")
            print(f"  User: {user_msg['content']}")
            print(f"  Expected: {expected_assistant_msg['content']}")
            
            # Get model's response
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7,
                )
                
                model_response = response.choices[0].message.content
                print(f"  Model: {model_response}")
                
                # Add model's response to messages for context in the next turn
                messages.append({"role": "assistant", "content": model_response})
                
                # Save result
                results.append({
                    "conversation_id": i,
                    "turn": j//2 + 1,
                    "user_message": user_msg["content"],
                    "expected_response": expected_assistant_msg["content"],
                    "model_response": model_response
                })
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results.append({
                    "conversation_id": i,
                    "turn": j//2 + 1,
                    "user_message": user_msg["content"],
                    "expected_response": expected_assistant_msg["content"],
                    "model_response": f"ERROR: {str(e)}"
                })
    
    return results

def save_results(results, output_file):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results (dict): Evaluation results
        output_file (str): Path to the output file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on multi-turn conversations")
    parser.add_argument("--model", required=True, help="Model to evaluate")
    parser.add_argument("--test_file", default="data/secret_word_dataset.json", help="Test dataset JSON file")
    parser.add_argument("--output_file", default="evaluation_results.json", help="Output file for evaluation results")
    parser.add_argument("--api_key", help="Together AI API key")
    
    args = parser.parse_args()
    
    # Load test conversations
    test_conversations = load_test_conversations(args.test_file)
    
    # Evaluate model
    results = evaluate_model(args.model, test_conversations, args.api_key)
    
    # Save results
    save_results(results, args.output_file)

if __name__ == "__main__":
    main() 