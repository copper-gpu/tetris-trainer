import sys
import subprocess
import textwrap

MENU = textwrap.dedent(
    """
    Tetris Trainer CLI
    ==================
    1. Play manually
    2. Train new model (fresh run)
    3. Resume training (continue)
    4. Exit
    """
)


def main():
    while True:
        print(MENU)
        choice = input("Select an option: ").strip()
        if choice == "1":
            subprocess.run([sys.executable, "play.py"], check=True)
        elif choice == "2":
            subprocess.run([sys.executable, "train_offline.py"], check=True)
        elif choice == "3":
            subprocess.run([sys.executable, "resume_training.py"], check=True)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.\n")


if __name__ == "__main__":
    main()
