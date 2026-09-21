"""Canonical PACS (Physics and Astronomy Classification Scheme, 2010) lookup.

Provides human-readable descriptions for PACS codes at three levels:
    division     -> first 2 digits, e.g. "74"        -> "Superconductivity"
    subdivision  -> "XX.YY",        e.g. "74.20"      -> (subdivision text if known, else division)
    specific     -> full code,      e.g. "74.20.-z"   -> (division/subdivision text + code)

Division-level (2-digit) descriptions are the authoritative anchor and cover the
whole scheme. Subdivision text is sparse on purpose — enrich SUBDIVISIONS over time
for the divisions that matter; describe() falls back to the division description.
"""

# 2-digit PACS divisions (PACS 2010 top level) -------------------------------
DIVISIONS = {
    "00": "General",
    "01": "Communication, education, history, and philosophy of physics",
    "02": "Mathematical methods in physics",
    "03": "Quantum mechanics, field theories, and special relativity",
    "04": "General relativity and gravitation",
    "05": "Statistical physics, thermodynamics, and nonlinear dynamical systems",
    "06": "Metrology, measurements, and laboratory procedures",
    "07": "Instruments, apparatus, and components common to several branches of physics and astronomy",
    "11": "General theory of fields and particles",
    "12": "Specific theories and interaction models; particle systematics",
    "13": "Specific reactions and phenomenology",
    "14": "Properties of specific particles",
    "21": "Nuclear structure",
    "23": "Radioactive decay and in-beam spectroscopy",
    "24": "Nuclear reactions: general",
    "25": "Nuclear reactions: specific reactions",
    "26": "Nuclear astrophysics",
    "27": "Properties of specific nuclei listed by mass ranges",
    "28": "Nuclear engineering and nuclear power studies",
    "29": "Experimental methods and instrumentation for elementary-particle and nuclear physics",
    "31": "Electronic structure of atoms and molecules: theory",
    "32": "Atomic properties and interactions with photons",
    "33": "Molecular properties and interactions with photons",
    "34": "Atomic and molecular collision processes and interactions",
    "36": "Exotic atoms and molecules; macromolecules; clusters",
    "37": "Mechanical control of atoms, molecules, and ions",
    "39": "Instrumentation and techniques for atomic and molecular physics",
    "41": "Electromagnetism; electron and ion optics",
    "42": "Optics",
    "43": "Acoustics",
    "44": "Heat transfer",
    "45": "Classical mechanics of discrete systems",
    "46": "Continuum mechanics of solids",
    "47": "Fluid dynamics",
    "51": "Physics of gases",
    "52": "Physics of plasmas and electric discharges",
    "61": "Structure of solids and liquids; crystallography",
    "62": "Mechanical and acoustical properties of condensed matter",
    "63": "Lattice dynamics",
    "64": "Equations of state, phase equilibria, and phase transitions",
    "65": "Thermal properties of condensed matter",
    "66": "Nonelectronic transport properties of condensed matter",
    "67": "Quantum fluids and solids",
    "68": "Surfaces and interfaces; thin films and nanosystems",
    "71": "Electronic structure of bulk materials",
    "72": "Electronic transport in condensed matter",
    "73": "Electronic structure and electrical properties of surfaces, interfaces, thin films, and low-dimensional structures",
    "74": "Superconductivity",
    "75": "Magnetic properties and materials",
    "76": "Magnetic resonances and relaxations in condensed matter; Mossbauer effect",
    "77": "Dielectrics, piezoelectrics, and ferroelectrics and their properties",
    "78": "Optical properties, condensed-matter spectroscopy and other interactions of radiation and particles with condensed matter",
    "79": "Electron and ion emission by liquids and solids; impact phenomena",
    "81": "Materials science",
    "82": "Physical chemistry and chemical physics",
    "83": "Rheology",
    "84": "Electronics; radiowave and microwave technology; direct energy conversion and storage",
    "85": "Electronic and magnetic devices; microelectronics",
    "87": "Biological and medical physics",
    "88": "Renewable energy resources and applications",
    "89": "Other areas of applied and interdisciplinary physics",
    "91": "Solid Earth physics",
    "92": "Hydrospheric and atmospheric geophysics",
    "93": "Geophysical observations, instrumentation, and techniques",
    "94": "Physics of the ionosphere and magnetosphere",
    "95": "Fundamental astronomy and astrophysics; instrumentation, techniques, and astronomical observations",
    "96": "Solar system; planetology",
    "97": "Stars",
    "98": "Stellar systems; interstellar medium; galactic and extragalactic objects and systems; the Universe",
}

# Sparse subdivision text (XX.YY). Enrich for high-traffic divisions as needed.
SUBDIVISIONS = {
    "03.65": "Quantum mechanics",
    "03.67": "Quantum information",
    "03.75": "Matter waves",
    "04.70": "Physics of black holes",
    "05.10": "Computational methods in statistical physics and nonlinear dynamics",
    "05.30": "Quantum statistical mechanics",
    "05.40": "Fluctuation phenomena, random processes, noise, and Brownian motion",
    "05.45": "Nonlinear dynamics and chaos",
    "05.70": "Thermodynamics",
    "11.25": "Strings and branes",
    "12.38": "Quantum chromodynamics",
    "42.50": "Quantum optics",
    "42.65": "Nonlinear optics",
    "64.60": "General studies of phase transitions",
    "71.10": "Theories and models of many-electron systems",
    "73.43": "Quantum Hall effects",
    "74.20": "Theories and models of superconducting state",
    "74.70": "Superconducting materials other than cuprates",
    "75.10": "General theory and models of magnetic ordering",
    "75.50": "Studies of specific magnetic materials",
    "78.67": "Optical properties of low-dimensional, mesoscopic, and nanoscale materials and structures",
}


def parse(code):
    """Split a PACS code into (division, subdivision) string keys.

    >>> parse("74.20.-z") -> ("74", "74.20")
    >>> parse("05.70")     -> ("05", "05.70")
    >>> parse("74")        -> ("74", None)
    """
    if code is None:
        return None, None
    code = str(code).strip()
    if not code or code.lower() == "none":
        return None, None
    parts = code.split(".")
    division = parts[0][:2]
    subdivision = f"{division}.{parts[1]}" if len(parts) >= 2 else None
    return division, subdivision


def describe(code, level):
    """Canonical description for a PACS code at the requested level.

    level in {"division", "subdivision", "specific"}.
    Falls back: subdivision text -> division text; specific -> subdivision/division + code.
    Returns None for the main level (handled by category_table titles) or unknown codes.
    """
    division, subdivision = parse(code)
    if division is None:
        return None

    div_text = DIVISIONS.get(division)
    if level == "division":
        return div_text or f"PACS {division}"

    if level == "subdivision":
        return SUBDIVISIONS.get(subdivision) or div_text or f"PACS {subdivision or division}"

    # specific
    base = SUBDIVISIONS.get(subdivision) or div_text or f"PACS {division}"
    return f"{base} (PACS {code})"
