# cli.py
from dag import ClassificationDAG
from utils.logging import setup_logging
from datetime import datetime

def main():
    # Initialize
    logger = setup_logging()
    dag = ClassificationDAG("./fine_tuned_model")
    
    print("Self-Healing Sentiment Analysis CLI")
    print("Type your text and press Enter. Type 'quit' to exit.\n")
    
    while True:
        user_input = input(">> Input text: ").strip()
        
        if user_input.lower() in ('quit', 'exit'):
            break
            
        # Run pipeline
        start_time = datetime.now()
        result = dag.run(user_input)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Display results
        if 'user_override' in result:
            print(f"\n✓ Final Label (user corrected): {result['user_override']}")
        else:
            print(f"\n✓ Final Label: {result['prediction']} ({result['confidence']:.1%})")
        
        print(f"⏱️  Processing time: {duration:.2f}s")
        print("---\n")

if __name__ == "__main__":
    main()