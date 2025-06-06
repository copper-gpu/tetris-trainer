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
    4. View best model (live)
    5. Exit
    """
)


def main():
    while True:
        print(MENU)
        choice = input("Select an option: ").strip()
        if choice == "1":
            subprocess.run([sys.executable, "play.py"], check=True)
        elif choice == "2":
            steps = input("Train for how many steps? [10_000_000]: ").strip()
            cmd = [sys.executable, "train_offline.py"]
            if steps:
                cmd += ["--steps", steps]
            subprocess.run(cmd, check=True)
        elif choice == "3":
            steps = input("Additional steps to train? [10_000_000]: ").strip()
            cmd = [sys.executable, "resume_training.py"]
            if steps:
                cmd += ["--steps", steps]
            subprocess.run(cmd, check=True)
        elif choice == "4":
            subprocess.run([sys.executable, "live_view.py"], check=True)
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.\n")


if __name__ == "__main__":
    main()
