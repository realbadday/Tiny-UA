#!/usr/bin/env python3
"""
Simple Training Launcher - Independent training interface
"""

import os
import sys

def show_training_menu():
    """Show training options menu"""
    print("\n🎓 TinyLlama Training Interface")
    print("=" * 40)
    print("Available training options:")
    print("1. Quick training with sample dataset")
    print("2. Full training options")
    print("3. View training help")
    print("4. Exit")
    print()
    
    while True:
        try:
            choice = input("Select option (1-4): ").strip()
            
            if choice == '1':
                print("\n🏃 Starting quick training...")
                cmd = "python3 train_tinyllama.py --dataset data/sample_dataset.csv --epochs 1 --test-prompts"
                print(f"Running: {cmd}")
                os.system(cmd)
                break
                
            elif choice == '2':
                print("\n🔧 Full training options:")
                os.system("python3 train_tinyllama.py --help")
                print("\n💡 Example: python3 train_tinyllama.py --dataset your_data.csv --epochs 3 --merge-weights")
                print("\nRun your custom training command directly in the terminal.")
                break
                
            elif choice == '3':
                print("\n📚 Training Help:")
                print("- Place your CSV dataset in the 'data/' folder")
                print("- CSV should have 'question' and 'answer' columns")
                print("- Use --epochs 1-5 for training (more = better but slower)")
                print("- Add --test-prompts to test after training")
                print("- Add --merge-weights for easier deployment")
                print("- See TRAINING_SETUP.md for detailed guide")
                print("\nPress Enter to continue...")
                input()
                continue
                
            elif choice == '4':
                print("Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    show_training_menu()
