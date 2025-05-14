from rich.console import Console
from rich.panel import Panel
from rich.text import Text
# Align is no longer needed for the main panel
# from rich.align import Align

# ASCII Art definitions for "HACKAGENT" (7 lines high)
# Using '|||', '///', '\\\', '___' for strokes, ' ' for spaces.

LETTER_H = [
    r"|||   |||",
    r"|||   |||",
    r"|||   |||",
    r"|||||||||",
    r"|||   |||",
    r"|||   |||",
    r"|||   |||",
]

LETTER_A = [
    r"  ///\\\    ",
    r" ///  \\\   ",
    r"///    \\\  ",
    r"||||||||||| ",
    r"|||     ||| ",
    r"|||     ||| ",
    r"|||     ||| ",
]

LETTER_C = [
    r"  ///////  ",
    r" ///       ",
    r"|||        ",
    r"|||        ",
    r"|||        ",
    r" \\\       ",
    r"  \\\\\\\  ",
]

LETTER_K = [
    r"|||   /// ",
    r"|||  ///  ",
    r"||| ///   ",
    r"||||||    ",
    r"||| \\\   ",
    r"|||  \\\  ",
    r"|||   \\\ ",
]

LETTER_G = [
    r"  ////////  ",
    r" ///        ",
    r"|||         ",
    r"|||    |||||",
    r"|||      |||",
    r" \\\    /// ",
    r"  \\\\////  ",
]

LETTER_E = [
    r"|||||||||",
    r"|||      ",
    r"|||      ",
    r"|||||||||",
    r"|||      ",
    r"|||      ",
    r"|||||||||",
]

LETTER_N = [
    r"|||     |||",
    r"||||\   |||",
    r"||| \\  |||",
    r"|||  \\ |||",
    r"|||   \\|||",
    r"|||    \|||",
    r"|||     |||",
]

LETTER_T = [
    r"|||||||||||",
    r"    |||    ",
    r"    |||    ",
    r"    |||    ",
    r"    |||    ",
    r"    |||    ",
    r"    |||    ",
]


# Map letters to their ASCII art
CHAR_MAP = {
    "H": LETTER_H,
    "A": LETTER_A,
    "C": LETTER_C,
    "K": LETTER_K,
    "G": LETTER_G,
    "E": LETTER_E,
    "N": LETTER_N,
    "T": LETTER_T,
    " ": [r"    "] * 7,  # Reduced space width (e.g., 4 spaces)
}


def generate_block_text(text: str, char_map: dict) -> str:
    """Generates a single multi-line string for the block text."""
    output_lines = [""] * 7  # Assuming all letters are 7 lines high
    letter_spacing = " "  # Reduced to one space between letters

    for i, char_in_text in enumerate(text.upper()):
        # Default to space art if char not in map (should not happen for HACKAGENT)
        char_art_lines = char_map.get(char_in_text, char_map[" "])
        for line_num in range(7):
            if (
                i > 0
            ):  # Add spacing BEFORE the character, but not for the first character
                output_lines[line_num] += letter_spacing
            output_lines[line_num] += char_art_lines[line_num]

    return "\n".join(output_lines)


def display_hackagent_splash():
    """Displays the HackAgent splash screen with HUGE block text, using slashes and more compact spacing."""
    console = Console()

    hack_text_str = generate_block_text("HACK", CHAR_MAP)
    agent_text_str = generate_block_text("AGENT", CHAR_MAP)

    full_block_text_str = f"{hack_text_str}\n\n{agent_text_str}"

    title_content = Text(full_block_text_str, style="bold dark_red")

    splash_panel = Panel(
        title_content,
        # title="[dim]Welcome to[/dim]", # Title removed by user previously
        border_style="red",
        padding=(2, 2),
        expand=False,
    )

    console.print(splash_panel)
    console.print()
