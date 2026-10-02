"""smartpiper: the live (paper) service for the smart ML stock-picker.

Runs beside the existing piedpiper system, never inside it: own folder (~/smartpiper on
the box), own venv, own systemd units. Research code in scripts/ is reused unchanged so
live decisions come from exactly the backtested maths.
"""
