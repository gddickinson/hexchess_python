# HexChess

![HexChess Logo](path/to/your/logo.png)

## Introduction

Welcome to HexChess, an innovative and challenging variant of chess played on a hexagonal board! This project brings together the strategic depth of traditional chess with the unique geometry of a hexagonal grid, creating a fresh and exciting gaming experience.

HexChess introduces new piece movements, tactical considerations, and strategic concepts that will intrigue both chess enthusiasts and casual players alike. With AI opponents of varying difficulty and a user-friendly interface, HexChess offers endless hours of entertainment and intellectual stimulation.

## Features

- 🔹 Hexagonal chess board with intuitive piece movements
- 🤖 Play against AI opponents with Basic and Advanced difficulty levels
- 🏆 Engage in AI vs. AI tournaments to study different strategies
- 📊 Real-time statistics and performance analysis
- 🎨 Clean and modern PyQt5-based user interface
- 📚 Comprehensive help system with game rules and strategy tips
- 💾 Save and load game functionality
- 🧠 Self-improving AI that learns from each game

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/hexchess.git
   ```

2. Navigate to the project directory:
   ```
   cd hexchess
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the game, run the following command from the project directory:

```
python hexchess_main.py
```

### Game Modes

1. **Player vs. AI**: Test your skills against the computer. Choose between Basic AI for a casual game or Advanced AI for a real challenge.

2. **Player vs. Player**: Enjoy a game of HexChess with a friend on the same computer.

3. **AI vs. AI Tournament**: Watch AIs battle it out and learn from their strategies. Customize tournament settings like the number of games and move delay.

### Controls

- Use the mouse to select and move pieces on the board.
- The status bar at the bottom of the window shows the current game state and whose turn it is.
- Access additional options and features through the menu bar at the top of the window.

## Rules

HexChess follows most of the traditional chess rules with adaptations for the hexagonal board:

- The game is played on a hexagonal board with 91 cells.
- Each player starts with 16 pieces: 1 King, 1 Queen, 2 Rooks, 2 Bishops, 2 Knights, 1 Dragon (unique to HexChess), and 7 Pawns.
- Piece movements are adapted to the hexagonal grid. For example, Rooks move in straight lines along the six hexagonal directions.
- Pawns move forward in two directions and capture diagonally.
- The Dragon combines the movements of the Queen and Knight.
- Check, checkmate, and stalemate rules apply similar to traditional chess.
- Pawns promote to any other piece (except King) upon reaching the opposite edge of the board.

For a detailed explanation of piece movements and rules, please refer to the in-game help section.

## Contributing

We welcome contributions to HexChess! If you have ideas for improvements or have found a bug, please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape HexChess
- Inspired by various hexagonal chess variants and traditional chess engines
- Built with PyQt5 and Python

---

Enjoy playing HexChess, and may the best strategist win! 🏆♟️
