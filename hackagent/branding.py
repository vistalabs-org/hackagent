from rich.console import Console
from rich.panel import Panel
from rich.text import Text
# Align is no longer needed for the main panel
# from rich.align import Align

# ASCII Art definitions for "HACKAGENT" (7 lines high)
# Using '|||', '///', '\\\\\\', '___' for strokes, ' ' for spaces.

HACKAGENT = """
██╗  ██╗ █████╗  ██████╗██╗  ██╗            
██║  ██║██╔══██╗██╔════╝██║ ██╔╝            
███████║███████║██║     █████╔╝             
██╔══██║██╔══██║██║     ██╔═██╗             
██║  ██║██║  ██║╚██████╗██║  ██╗            
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝            
                                            
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                                               
"""


def display_hackagent_splash():
    """Displays the HackAgent splash screen using the pre-defined ASCII art."""
    console = Console()

    # Create a Text object from the HACKAGENT string
    title_content = Text(HACKAGENT, style="bold dark_red")

    splash_panel = Panel(
        title_content,
        border_style="red",
        padding=(2, 2),
        expand=False,
    )

    console.print(splash_panel)
    console.print()
