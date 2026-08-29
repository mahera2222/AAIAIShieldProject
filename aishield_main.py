import subprocess
import sys


def main():
    print("AIShield - Image Tampering Detection")
    print("Launching Streamlit application...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py"
        ],
        check=True
    )


if __name__ == "__main__":
    main()
