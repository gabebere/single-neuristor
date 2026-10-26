# Quick setup for Amir

Hi Amir — quick setup instructions:

1. Copy the shared `single-neuristor` folder onto your computer. Keep
   `Simulations_on_VO2`, `src`, and the supplied `public_jobs` folders together—the
   HTML viewers need their GIF files.

2. Open Terminal inside `single-neuristor` and check Python:

   ```bash
   python3 --version
   ```

   For viewing the presentation, Python **3.9 or newer** is enough. No additional
   Python packages or LaTeX installation are required.

3. Start everything:

   ```bash
   python3 src/neuristor/interfaces/presentation_server.py --present
   ```

   This opens the PDF and starts the interactive viewers. Click the buttons in the
   slides to open them. Everything runs locally, without internet. Run the same
   command again after restarting the computer.

If you also want to run **new simulations**, use Python **3.10 or newer** and
install the research dependencies once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For future simulation sessions, activate the environment with
`source .venv/bin/activate`.
