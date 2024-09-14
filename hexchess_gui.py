import sys
import random
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QAction, QMenuBar, QDialog, QCheckBox, QGridLayout, QFileDialog, QFrame,
                             QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QTabWidget, QMessageBox)
from PyQt5.QtGui import QPainter, QColor, QPen, QPolygonF, QBrush, QFont, QPixmap
from PyQt5.QtCore import Qt, QPointF, QTimer
from hexchess_game import HexChess
from hexchess_ai import BasicAI, AdvancedAI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
from typing import List, Tuple, Optional, Dict, Any
from hexchess_testing import TestingWindow
from hexchess_help import HelpWindow
import logging

logger = logging.getLogger('hexchess')

class BoardSizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Board Size")
        layout = QVBoxLayout()

        self.size_spinner = QSpinBox()
        self.size_spinner.setRange(3, 10)  # Adjust range as needed
        self.size_spinner.setValue(8)  # Default size

        layout.addWidget(QLabel("Board Size:"))
        layout.addWidget(self.size_spinner)

        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)

        layout.addLayout(buttons)
        self.setLayout(layout)


class GameSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Game Setup")
        self.setGeometry(300, 300, 300, 200)

        layout = QVBoxLayout()

        self.white_player = QComboBox()
        self.white_player.addItems(["Human", "BasicAI", "AdvancedAI"])  # Add AdvancedAI option
        layout.addWidget(QLabel("White Player:"))
        layout.addWidget(self.white_player)

        self.black_player = QComboBox()
        self.black_player.addItems(["Human", "BasicAI", "AdvancedAI"])  # Add AdvancedAI option
        layout.addWidget(QLabel("Black Player:"))
        layout.addWidget(self.black_player)

        self.start_button = QPushButton("Start Game")
        self.start_button.clicked.connect(self.accept)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

class TournamentSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tournament Setup")
        self.setGeometry(300, 300, 300, 300)

        layout = QVBoxLayout()

        self.num_games = QSpinBox()
        self.num_games.setRange(1, 100)
        self.num_games.setValue(10)
        layout.addWidget(QLabel("Number of Games:"))
        layout.addWidget(self.num_games)

        self.ai_player1 = QComboBox()
        self.ai_player1.addItems(["BasicAI", "AdvancedAI"])  # Add AdvancedAI option
        layout.addWidget(QLabel("AI Player 1:"))
        layout.addWidget(self.ai_player1)

        self.ai_player2 = QComboBox()
        self.ai_player2.addItems(["BasicAI", "AdvancedAI"])  # Add AdvancedAI option
        layout.addWidget(QLabel("AI Player 2:"))
        layout.addWidget(self.ai_player2)

        self.visual_mode = QCheckBox("Visual Representation")
        self.visual_mode.setChecked(True)
        layout.addWidget(self.visual_mode)

        self.move_delay = QSpinBox()
        self.move_delay.setRange(0, 5000)
        self.move_delay.setValue(500)
        self.move_delay.setSuffix(" ms")
        layout.addWidget(QLabel("Move Delay (Visual Mode):"))
        layout.addWidget(self.move_delay)

        self.start_button = QPushButton("Start Tournament")
        self.start_button.clicked.connect(self.accept)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

class TournamentResultsDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tournament Results")
        self.setGeometry(300, 300, 400, 300)

        layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Game", "White", "Black", "Result"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i, result in enumerate(results):
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.results_table.setItem(i, 1, QTableWidgetItem(result['white']))
            self.results_table.setItem(i, 2, QTableWidgetItem(result['black']))
            self.results_table.setItem(i, 3, QTableWidgetItem(result['result']))

        layout.addWidget(self.results_table)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)

        self.setLayout(layout)

class StatsWindow(QFrame):
    def __init__(self, player_color, parent=None):
        super().__init__(parent)
        self.player_color = player_color
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)

        layout = QVBoxLayout()
        self.title_label = QLabel(f"{'White' if player_color == 'w' else 'Black'} Player Stats")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont('Arial', 12, QFont.Bold))
        layout.addWidget(self.title_label)

        self.stats_labels = {
            'material': QLabel("Material: 0"),
            'center_control': QLabel("Center Control: 0"),
            'king_safety': QLabel("King Safety: 0"),
            'mobility': QLabel("Mobility: 0"),
            'pawn_structure': QLabel("Pawn Structure: 0")
        }

        for label in self.stats_labels.values():
            layout.addWidget(label)

        self.setLayout(layout)

    def update_stats(self, stats):
        self.stats_labels['material'].setText(f"Material: {stats['material_balance']}")
        self.stats_labels['center_control'].setText(f"Center Control: {stats['center_control']:.2f}")
        self.stats_labels['king_safety'].setText(f"King Safety: {stats['king_safety']:.2f}")
        self.stats_labels['mobility'].setText(f"Mobility: {stats['mobility']}")
        self.stats_labels['pawn_structure'].setText(f"Pawn Structure: {stats['pawn_structure']:.2f}")


class OptionsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Game Options")
        self.setGeometry(200, 200, 300, 200)

        layout = QGridLayout()

        self.show_coordinates = QCheckBox("Show Coordinates")
        self.show_possible_moves = QCheckBox("Show Possible Moves")
        self.use_piece_images = QCheckBox("Use Piece Images")

        # Set default checked state
        self.show_possible_moves.setChecked(True)
        self.use_piece_images.setChecked(True)

        layout.addWidget(self.show_coordinates, 0, 0)
        layout.addWidget(self.show_possible_moves, 1, 0)
        layout.addWidget(self.use_piece_images, 2, 0)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_button, 3, 0)

        self.setLayout(layout)

    def apply_settings(self):
        self.parent().apply_options(
            self.show_coordinates.isChecked(),
            self.show_possible_moves.isChecked(),
            self.use_piece_images.isChecked()
        )
        self.close()

class HexBoardWidget(QWidget):
    def __init__(self, game: HexChess, parent=None):
        super().__init__(parent)
        self.game = game
        self.selected_cell = None
        self.setMouseTracking(True)
        self.main_window = None
        self.show_possible_moves = True
        self.use_piece_images = True
        self.piece_images = {}
        self.load_piece_images()

    def load_piece_images(self):
        pieces = ['P', 'R', 'N', 'B', 'Q', 'K', 'D', 'C']
        colors = ['w', 'b']
        for color in colors:
            for piece in pieces:
                img_path = f"images/{color}{piece}.png"
                if os.path.exists(img_path):
                    self.piece_images[f"{color}{piece}"] = QPixmap(img_path)
                else:
                    print(f"Warning: Image not found for {color}{piece}")


    def set_main_window(self, main_window):
        self.main_window = main_window

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        self.calculate_hex_size()
        self.calculate_board_offset()

        for q, r in self.game.board.get_all_cells():
            self._draw_hex(painter, q, r)

    def calculate_hex_size(self):
        board_diameter = 2 * self.game.board.size + 1
        width_size = self.width() / (board_diameter * math.sqrt(3))
        height_size = self.height() / (board_diameter * 1.5)
        self.hex_size = min(width_size, height_size) * 0.95

    def calculate_board_offset(self):
        board_width = (2 * self.game.board.size + 1) * self.hex_size * math.sqrt(3)
        board_height = (2 * self.game.board.size + 1) * self.hex_size * 1.5
        self.offset_x = (self.width() - board_width) / 2 + board_width / 2
        self.offset_y = (self.height() - board_height) / 2 + board_height / 2

    def _draw_hex(self, painter: QPainter, q: int, r: int):
        x, y = self._hex_to_pixel(q, r)

        color = QColor(240, 240, 240) if (q + r) % 2 == 0 else QColor(200, 200, 200)
        if (q, r) == self.selected_cell:
            color = QColor(255, 255, 0)  # Yellow for selected cell
        elif self.show_possible_moves and self.selected_cell:
            if (q, r) in self.game.get_possible_moves(*self.selected_cell):
                color = QColor(0, 255, 0, 100)  # Light green for possible moves

        painter.setBrush(QBrush(color))
        painter.setPen(QPen(Qt.black, 1))

        points = QPolygonF([QPointF(x + self.hex_size * math.cos(math.radians(angle)),
                                    y + self.hex_size * math.sin(math.radians(angle)))
                            for angle in range(30, 390, 60)])
        painter.drawPolygon(points)

        piece = self.game.board.get_cell(q, r)
        if piece:
            if self.use_piece_images and piece in self.piece_images:
                img = self.piece_images[piece].scaled(int(self.hex_size), int(self.hex_size),
                                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap(int(x - self.hex_size/2), int(y - self.hex_size/2), img)
            else:
                painter.setFont(QFont('Arial', int(self.hex_size / 2)))
                painter.drawText(QPointF(x - self.hex_size / 4, y + self.hex_size / 4), piece)

    def _hex_to_pixel(self, q: int, r: int):
        x = self.hex_size * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
        y = self.hex_size * (3/2 * r)
        return x + self.offset_x, y + self.offset_y

    def _pixel_to_hex(self, x: float, y: float):
        x = (x - self.offset_x) / self.hex_size
        y = (y - self.offset_y) / self.hex_size
        q = (math.sqrt(3)/3 * x - 1/3 * y)
        r = (2/3 * y)
        return self._hex_round(q, r)

    def _hex_round(self, q: float, r: float):
        s = -q - r
        rq = round(q)
        rr = round(r)
        rs = round(s)
        q_diff = abs(rq - q)
        r_diff = abs(rr - r)
        s_diff = abs(rs - s)
        if q_diff > r_diff and q_diff > s_diff:
            rq = -rr - rs
        elif r_diff > s_diff:
            rr = -rq - rs
        else:
            rs = -rq - rr
        return rq, rr

    def mousePressEvent(self, event):
        q, r = self._pixel_to_hex(event.x(), event.y())
        if self.game.board.is_valid_cell(q, r):
            if self.selected_cell:
                self.main_window.make_move(self.selected_cell[0], self.selected_cell[1], q, r)
                self.selected_cell = None
            else:
                piece = self.game.board.get_cell(q, r)
                if piece and piece[0] == self.game.current_player:
                    self.selected_cell = (q, r)
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.calculate_hex_size()
        self.calculate_board_offset()
        self.update()


class LearningStatsWindow(QMainWindow):
    def __init__(self, ai: BasicAI):
        super().__init__()
        self.ai = ai
        self.setWindowTitle("AI Learning Statistics")
        self.setGeometry(100, 100, 800, 600)  # Increased size for more information
        central_widget = QWidget()
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_general_stats_widget(), "General Stats")
        self.tabs.addTab(self._create_params_widget(), "Parameters")
        self.tabs.addTab(self._create_charts_widget(), "Learning Charts")
        self.tabs.addTab(self._create_performance_widget(), "Recent Performance")
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _create_general_stats_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.games_played_label = QLabel(f"Games Played: {self.ai.learning_module.games_played}")
        self.win_rate_label = QLabel(f"Win Rate: {self.ai.learning_module.wins / max(1, self.ai.learning_module.games_played):.2%}")
        self.avg_moves_label = QLabel(f"Average Moves per Game: {self.ai.total_moves / max(1, self.ai.games_played):.2f}")
        layout.addWidget(self.games_played_label)
        layout.addWidget(self.win_rate_label)
        layout.addWidget(self.avg_moves_label)
        widget.setLayout(layout)
        return widget

    def _create_params_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(2)
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self._update_params_table()
        layout.addWidget(self.params_table)
        widget.setLayout(layout)
        return widget

    def _create_charts_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.chart_canvas = FigureCanvas(figure)
        self._update_charts(ax1, ax2)
        layout.addWidget(self.chart_canvas)
        widget.setLayout(layout)
        return widget

    def _create_performance_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.performance_table = QTableWidget()
        self.performance_table.setColumnCount(6)
        self.performance_table.setHorizontalHeaderLabels(["Game", "Result", "Material", "Center Control", "King Safety", "Mobility"])
        self._update_performance_table()
        layout.addWidget(self.performance_table)
        widget.setLayout(layout)
        return widget

    def _update_params_table(self):
        self.params_table.setRowCount(0)
        for param, value in self.ai.learning_module.params.items():
            if isinstance(value, dict):
                for sub_param, sub_value in value.items():
                    row = self.params_table.rowCount()
                    self.params_table.insertRow(row)
                    self.params_table.setItem(row, 0, QTableWidgetItem(f"{param} - {sub_param}"))
                    self.params_table.setItem(row, 1, QTableWidgetItem(f"{sub_value:.4f}"))
            else:
                row = self.params_table.rowCount()
                self.params_table.insertRow(row)
                self.params_table.setItem(row, 0, QTableWidgetItem(param))
                self.params_table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))

    def _update_charts(self, ax1, ax2):
        history = self.ai.learning_module.learning_history
        games = [entry['games_played'] for entry in history]
        win_rates = [entry['win_rate'] for entry in history]

        ax1.clear()
        ax1.plot(games, win_rates)
        ax1.set_xlabel('Games Played')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate Progress')

        # Plot parameter changes over time
        ax2.clear()
        for param, values in self.ai.learning_module.params.items():
            if not isinstance(values, dict):
                param_history = [entry['params'][param] for entry in history]
                ax2.plot(games, param_history, label=param)
        ax2.set_xlabel('Games Played')
        ax2.set_ylabel('Parameter Value')
        ax2.set_title('Parameter Changes')
        ax2.legend()

        self.chart_canvas.draw()

    def _update_performance_table(self):
        self.performance_table.setRowCount(0)
        for i, perf in enumerate(self.ai.performance_history[-20:]):  # Show last 20 games
            row = self.performance_table.rowCount()
            self.performance_table.insertRow(row)
            self.performance_table.setItem(row, 0, QTableWidgetItem(str(perf['game_number'])))
            self.performance_table.setItem(row, 1, QTableWidgetItem(f"{perf['result']:.2f}"))
            self.performance_table.setItem(row, 2, QTableWidgetItem(f"{perf['material_balance']:.2f}"))
            self.performance_table.setItem(row, 3, QTableWidgetItem(f"{perf['center_control']:.2f}"))
            self.performance_table.setItem(row, 4, QTableWidgetItem(f"{perf['king_safety']:.2f}"))
            self.performance_table.setItem(row, 5, QTableWidgetItem(f"{perf['mobility']:.2f}"))

    def update_stats(self):
        self.games_played_label.setText(f"Games Played: {self.ai.learning_module.games_played}")
        self.win_rate_label.setText(f"Win Rate: {self.ai.learning_module.wins / max(1, self.ai.learning_module.games_played):.2%}")
        self.avg_moves_label.setText(f"Average Moves per Game: {self.ai.total_moves / max(1, self.ai.games_played):.2f}")
        self._update_params_table()
        self._update_charts(self.chart_canvas.figure.axes[0], self.chart_canvas.figure.axes[1])
        self._update_performance_table()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hexagonal Chess")
        self.setGeometry(100, 100, 1200, 800)

        self.game = HexChess()
        self.white_player = "Human"
        self.black_player = "Human"
        self.white_ai = None
        self.black_ai = None
        self.ai_move_timer = QTimer(self)
        self.ai_move_timer.timeout.connect(self.make_ai_move)

        self.tournament_timer = QTimer()
        self.tournament_timer.timeout.connect(self.make_tournament_move)

        self.tournament_in_progress = False
        self.tournament_games_left = 0
        self.tournament_results = []
        self.tournament_ai_players = None
        self.tournament_paused = False

        self.learning_stats_window = None
        self.help_window = None

        self.create_menu_bar()

        central_widget = QWidget()
        main_layout = QHBoxLayout()

        self.white_stats = StatsWindow('w')
        main_layout.addWidget(self.white_stats)

        game_layout = QVBoxLayout()
        self.board_widget = HexBoardWidget(self.game, self)
        self.board_widget.set_main_window(self)
        game_layout.addWidget(self.board_widget, 1)

        self.status_label = QLabel("White's turn")
        self.status_label.setAlignment(Qt.AlignCenter)
        game_layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset Game")
        self.reset_button.clicked.connect(self.reset_game)
        button_layout.addWidget(self.reset_button)

        self.pause_button = QPushButton("Pause Tournament")
        self.pause_button.clicked.connect(self.toggle_tournament_pause)
        self.pause_button.setEnabled(False)
        button_layout.addWidget(self.pause_button)

        self.exit_stuck_button = QPushButton("Exit Stuck Game")
        self.exit_stuck_button.clicked.connect(self.exit_stuck_game)
        self.exit_stuck_button.setEnabled(False)
        button_layout.addWidget(self.exit_stuck_button)

        game_layout.addLayout(button_layout)

        main_layout.addLayout(game_layout, 2)

        self.black_stats = StatsWindow('b')
        main_layout.addWidget(self.black_stats)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)



    def setup_new_tournament(self):
        dialog = TournamentSetupDialog(self)
        if dialog.exec_():
            self.num_games = dialog.num_games.value()
            ai_player1 = self.create_player(dialog.ai_player1.currentText(), 'w')
            ai_player2 = self.create_player(dialog.ai_player2.currentText(), 'b')
            visual_mode = dialog.visual_mode.isChecked()
            move_delay = dialog.move_delay.value()

            # Store the AI players
            self.tournament_ai_players = (ai_player1, ai_player2)

            self.run_tournament(self.num_games, ai_player1, ai_player2, visual_mode, move_delay)


    def create_player(self, player_type, color):
        if player_type == "Human":
            return None
        elif player_type == "BasicAI":
            return BasicAI(self.game, color)
        elif player_type == "AdvancedAI":
            return AdvancedAI(self.game, color)
        else:
            raise ValueError("Unknown Player type")

    def run_tournament(self, num_games, ai_player1, ai_player2, visual_mode, move_delay):
        self.tournament_in_progress = True
        self.tournament_games_left = num_games
        self.tournament_results = []
        self.white_player, self.black_player = ai_player1, ai_player2
        self.tournament_paused = False

        if visual_mode:
            self.tournament_timer.setInterval(move_delay)
            self.tournament_timer.start()
            self.pause_button.setEnabled(True)
            self.pause_button.setText("Pause Tournament")
            self.exit_stuck_button.setEnabled(True)
        else:
            while self.tournament_games_left > 0:
                self.play_full_game()
                self.save_ai_progress()

        if not visual_mode:
            self.end_tournament()

    def exit_stuck_game(self):
        if self.tournament_in_progress:
            logging.info("Exiting stuck game and penalizing AI parameters")
            self.penalize_ai_parameters()
            self.end_current_game()
            if self.tournament_games_left > 0:
                self.start_new_game()
            else:
                self.end_tournament()

    def penalize_ai_parameters(self):
        penalty_factor = 0.9  # Reduce parameters by 10%
        if isinstance(self.white_player, BasicAI):
            self.white_player.penalize_parameters(penalty_factor)
        if isinstance(self.black_player, BasicAI):
            self.black_player.penalize_parameters(penalty_factor)


    def make_tournament_move(self):
        if self.tournament_paused:
            return

        if self.game.is_game_over():
            self.end_current_game()
            if self.tournament_games_left > 0:
                self.start_new_game()
            else:
                self.end_tournament()
        else:
            current_player = self.white_player if self.game.current_player == 'w' else self.black_player
            move = current_player.make_move()
            if move:
                self.game.make_move(*move)
                self.board_widget.update()
                self.update_status()
                self.update_learning_stats()

                # Check if the game ended after this move
                if self.game.is_game_over():
                    self.end_current_game()
                    if self.tournament_games_left > 0:
                        self.start_new_game()
                    else:
                        self.end_tournament()

    def play_full_game(self):
        self.reset_game()
        while not self.game.is_game_over():
            if self.game.current_player == 'w':
                move = self.white_player.make_move()
            else:
                move = self.black_player.make_move()

            if move:
                self.game.make_move(*move)

        self.end_current_game()

    def end_current_game(self):
        winner = self.game.get_winner()
        result = 1 if winner == 'w' else 0 if winner == 'b' else 0.5

        # Update AI learning
        if isinstance(self.white_player, BasicAI):
            self.white_player.learn_from_game(result, self.get_game_stats())
        if isinstance(self.black_player, BasicAI):
            self.black_player.learn_from_game(1 - result, self.get_game_stats())

        self.tournament_results.append({
            'white': type(self.white_player).__name__,
            'black': type(self.black_player).__name__,
            'result': 'White' if winner == 'w' else 'Black' if winner == 'b' else 'Draw'
        })
        self.tournament_games_left -= 1
        self.save_ai_progress()
        self.update_learning_stats()  # Update learning stats after each game


    def start_new_game(self):
        self.reset_game()
        self.white_player, self.black_player = self.black_player, self.white_player
        self.white_player.color, self.black_player.color = 'w', 'b'
        self.white_player.game = self.black_player.game = self.game
        self.board_widget.update()
        self.update_status()

    def end_tournament(self):
        self.tournament_in_progress = False
        if hasattr(self, 'tournament_timer'):
            self.tournament_timer.stop()
        self.save_ai_progress()  # Final save at the end of the tournament
        self.show_tournament_results()
        self.update_learning_stats()

        # Update learning stats if window is open
        if hasattr(self, 'learning_stats_window') and self.learning_stats_window.isVisible():
            self.learning_stats_window.update_stats()

        # Clear the tournament AI players
        self.tournament_ai_players = None

        self.exit_stuck_button.setEnabled(False)  # Disable the button after tournament
        self.pause_button.setEnabled(False)



    def update_status(self):
        if self.tournament_in_progress:
            self.status_label.setText(f"Tournament in progress. Games left: {self.tournament_games_left}")
        elif self.game.is_game_over():
            winner = self.game.get_winner()
            if winner == 'draw':
                self.status_label.setText("Game Over! It's a draw!")
            else:
                self.status_label.setText(f"Game Over! {'White' if winner == 'w' else 'Black'} wins!")
        else:
            self.status_label.setText("White's turn" if self.game.current_player == 'w' else "Black's turn")


    def show_tournament_results(self):
        dialog = TournamentResultsDialog(self.tournament_results, self)
        dialog.exec_()



    def make_move(self, from_q: int, from_r: int, to_q: int, to_r: int):
        if self.game.make_move(from_q, from_r, to_q, to_r):
            self.board_widget.update()
            self.update_status()
            self.update_stats()

            if self.game.is_game_over():
                self.handle_game_over()
            else:
                self.make_ai_move_if_needed()


    def make_ai_move_if_needed(self):
        current_player = self.game.current_player
        ai = self.white_ai if current_player == 'w' else self.black_ai

        if ai:
            ai_move = ai.make_move()
            if ai_move:
                self.game.make_move(*ai_move)
                self.board_widget.update()
                self.update_status()
                self.update_stats()

                if not self.game.is_game_over():
                    # If it's still AI's turn (in case of AI vs AI), schedule next move
                    if (current_player == 'w' and self.black_player == "AI") or \
                       (current_player == 'b' and self.white_player == "AI"):
                        self.ai_move_timer.start(1000)
                else:
                    self.ai_move_timer.stop()

    def make_ai_move(self):
        self.make_ai_move_if_needed()

    def reset_game(self):
        self.game = HexChess()
        if self.white_ai:
            self.white_ai.game = self.game
        if self.black_ai:
            self.black_ai.game = self.game
        self.board_widget.game = self.game
        self.board_widget.selected_cell = None
        self.board_widget.update()
        self.update_status()
        self.update_stats()
        self.ai_move_timer.stop()

        # Update the learning stats window if it exists
        if self.learning_stats_window and self.learning_stats_window.isVisible():
            if self.white_ai:
                self.learning_stats_window.ai = self.white_ai
            elif self.black_ai:
                self.learning_stats_window.ai = self.black_ai
            self.learning_stats_window.update_stats()

        # Start the game if it's AI vs AI
        if self.white_player == "AI" and self.black_player == "AI":
            self.ai_move_timer.start(1000)


    def play_full_game_with_human(self):
        self.reset_game()
        while not self.game.is_game_over():

            if self.game.current_player == 'w':
                if self.white_player == "AI":
                    move = self.white_player.make_move()
            else:
                if self.black_player == "AI":
                    move = self.black_player.make_move()

            if move:
                self.game.make_move(*move)

        self.end_current_game()

    def setup_new_game(self):
        dialog = GameSetupDialog(self)
        if dialog.exec_():
            self.white_player = dialog.white_player.currentText()
            self.black_player = dialog.black_player.currentText()

            print(f'White player = {self.white_player}')
            print(f'Black player = {self.black_player}')

            if self.white_player == "BasicAI":
                self.white_ai = BasicAI(self.game, 'w')
                self.white_ai.load_learning_progress()
            elif self.white_player == "AdvancedAI":
                self.white_ai = AdvancedAI(self.game, 'w')
                self.white_ai.load_learning_progress()
            else:
                self.white_ai = None

            if self.black_player == "BasicAI":
                self.black_ai = BasicAI(self.game, 'b')
                self.black_ai.load_learning_progress()
            elif self.black_player == "AdvancedAI":
                self.black_ai = AdvancedAI(self.game, 'b')
                self.black_ai.load_learning_progress()
            else:
                self.black_ai = None

            self.reset_game()
            self.status_label.setText("White's turn")
            self.update_stats()
            self.board_widget.update()

            # Update the learning stats window if it exists
            if self.learning_stats_window:
                if self.white_ai:
                    self.learning_stats_window.ai = self.white_ai
                elif self.black_ai:
                    self.learning_stats_window.ai = self.black_ai
                self.learning_stats_window.update_stats()

            # Start the game if it's AI vs AI
            if 'AI' in self.white_player and 'AI' in self.black_player:
                self.ai_move_timer.start(1000)


    def create_menu_bar(self):
        menu_bar = self.menuBar()

        game_menu = menu_bar.addMenu("Game")
        new_game_action = QAction("New Game", self)
        new_game_action.triggered.connect(self.setup_new_game)
        game_menu.addAction(new_game_action)

        set_board_size_action = QAction("Set Board Size", self)
        set_board_size_action.triggered.connect(self.set_board_size)
        game_menu.addAction(set_board_size_action)

        tournament_menu = menu_bar.addMenu("Tournament")
        new_tournament_action = QAction("New Tournament", self)
        new_tournament_action.triggered.connect(self.setup_new_tournament)
        tournament_menu.addAction(new_tournament_action)

        ai_menu = menu_bar.addMenu("AI")
        show_learning_stats_action = QAction("Show Learning Stats", self)
        show_learning_stats_action.triggered.connect(self.show_learning_stats)
        ai_menu.addAction(show_learning_stats_action)

        options_menu = menu_bar.addMenu("Options")
        open_options = QAction("Open Options", self)
        open_options.triggered.connect(self.open_options_window)
        options_menu.addAction(open_options)


        help_menu = menu_bar.addMenu("Help")
        show_help_action = QAction("Show Help", self)
        show_help_action.triggered.connect(self.show_help)
        help_menu.addAction(show_help_action)

    def set_board_size(self):
        dialog = BoardSizeDialog(self)
        if dialog.exec_():
            new_size = dialog.size_spinner.value()
            self.game = HexChess(board_size=new_size)
            self.board_widget.game = self.game
            self.board_widget.update()
            self.reset_game()

    def toggle_tournament_pause(self):
        if self.tournament_in_progress:
            if self.tournament_paused:
                self.tournament_paused = False
                self.pause_button.setText("Pause Tournament")
                self.tournament_timer.start()
            else:
                self.tournament_paused = True
                self.pause_button.setText("Resume Tournament")
                self.tournament_timer.stop()


    def open_testing_window(self):
        self.testing_window = TestingWindow()
        self.testing_window.show()


    def show_help(self):
        if not self.help_window:
            self.help_window = HelpWindow()
        self.help_window.show()


    def show_learning_stats(self):
        ai = None
        if self.tournament_in_progress and self.tournament_ai_players:
            # If a tournament is in progress, use the white player (arbitrarily chosen)
            ai = self.tournament_ai_players[0]
        elif hasattr(self, 'white_ai') and isinstance(self.white_ai, BasicAI):
            ai = self.white_ai
        elif hasattr(self, 'black_ai') and isinstance(self.black_ai, BasicAI):
            ai = self.black_ai

        if ai:
            if not self.learning_stats_window:
                self.learning_stats_window = LearningStatsWindow(ai)
            elif not self.learning_stats_window.isVisible():
                self.learning_stats_window.close()
                self.learning_stats_window = LearningStatsWindow(ai)
            self.learning_stats_window.show()
            self.learning_stats_window.update_stats()
            logger.info("Learning stats window opened")
        else:
            QMessageBox.information(self, "No AI", "No AI player is currently active.")
            logger.warning("Attempted to show learning stats, but no AI is active")


    def open_options_window(self):
        options_window = OptionsWindow(self)
        options_window.show()

    def apply_options(self, show_coordinates, show_possible_moves, use_piece_images):
        self.board_widget.show_coordinates = show_coordinates
        self.board_widget.show_possible_moves = show_possible_moves
        self.board_widget.use_piece_images = use_piece_images
        self.board_widget.update()

    def update_learning_stats(self):
        if hasattr(self, 'learning_stats_window') and self.learning_stats_window is not None and self.learning_stats_window.isVisible():
            self.learning_stats_window.update_stats()

    def get_game_stats(self) -> Dict[str, Any]:
        stats = {
            'total_moves': len(self.game.move_history),
            'material_balance': self._calculate_material_balance(),
            'center_control': self._calculate_center_control(),
            'king_safety': self._calculate_king_safety(),
            'mobility': self._calculate_mobility(),
            'pawn_structure': self._calculate_pawn_structure(),
            'piece_effectiveness': self._calculate_piece_effectiveness()
        }
        return stats

    def _calculate_material_balance(self) -> int:
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0, 'D': 15, 'C': 2}
        balance = 0
        for q, r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(q, r)
            if piece:
                value = piece_values[piece[1]]
                balance += value if piece[0] == 'w' else -value
        return balance

    def _calculate_center_control(self) -> float:
        center_hexes = [(0, 0), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]
        white_control = 0
        black_control = 0
        for q, r in center_hexes:
            white_control += len([move for move in self._get_all_moves('w') if move[2] == q and move[3] == r])
            black_control += len([move for move in self._get_all_moves('b') if move[2] == q and move[3] == r])
        total_control = white_control + black_control
        return (white_control - black_control) / total_control if total_control > 0 else 0

    def _calculate_king_safety(self) -> float:
        def king_safety_score(color):
            king_pos = self.find_king(color)
            if not king_pos:
                return 0
            q, r = king_pos
            safety = 0
            for dq, dr in [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]:
                adj_q, adj_r = q + dq, r + dr
                if self.game.board.is_valid_cell(adj_q, adj_r):
                    piece = self.game.board.get_cell(adj_q, adj_r)
                    if piece and piece[0] == color:
                        safety += 1
            return safety / 6  # Normalize to 0-1 range

        white_safety = king_safety_score('w')
        black_safety = king_safety_score('b')
        return white_safety - black_safety

    def _calculate_mobility(self) -> int:
        white_mobility = len(self._get_all_moves('w'))
        black_mobility = len(self._get_all_moves('b'))
        return white_mobility - black_mobility

    def _calculate_pawn_structure(self) -> float:
        def pawn_structure_score(color):
            score = 0
            pawns = [(q, r) for q, r in self.game.board.get_all_cells()
                     if self.game.board.get_cell(q, r) == f'{color}P']
            for q, r in pawns:
                if self.is_passed_pawn(q, r, color):
                    score += 2
                if self.is_connected_pawn(q, r, color):
                    score += 1
            return score

        white_score = pawn_structure_score('w')
        black_score = pawn_structure_score('b')
        return white_score - black_score

    def _calculate_piece_effectiveness(self) -> Dict[str, float]:
        effectiveness = {piece: 0 for piece in 'PRNBQKDC'}
        total_moves = len(self.game.move_history)
        if total_moves == 0:
            return effectiveness

        for from_q, from_r, to_q, to_r, captured in self.game.move_history:
            piece = self.game.board.get_cell(to_q, to_r)
            if piece:
                effectiveness[piece[1]] += 1
                if captured:
                    effectiveness[piece[1]] += 2  # Bonus for capturing

        # Normalize effectiveness
        for piece in effectiveness:
            effectiveness[piece] /= total_moves

        return effectiveness

    def update_stats(self):
        game_stats = self.calculate_game_statistics()
        self.white_stats.update_stats(game_stats['w'])
        self.black_stats.update_stats(game_stats['b'])

    def handle_game_over(self):
        winner = self.game.get_winner()
        if winner == 'draw':
            result = 0.5
        else:
            result = 1 if winner == 'w' else 0

        game_stats = self.calculate_game_statistics()

        if self.white_ai:
            self.white_ai.learn_from_game(result, game_stats['w'])
            self.white_ai.save_learning_progress()
            logger.info("White AI learning progress saved after game")

        if self.black_ai:
            self.black_ai.learn_from_game(1 - result, game_stats['b'])
            self.black_ai.save_learning_progress()
            logger.info("Black AI learning progress saved after game")


        if self.learning_stats_window and self.learning_stats_window.isVisible():
            self.learning_stats_window.update_stats()


    def calculate_game_statistics(self):
        stats = {
            'w': {'material_balance': 0, 'center_control': 0, 'king_safety': 0, 'mobility': 0, 'pawn_structure': 0},
            'b': {'material_balance': 0, 'center_control': 0, 'king_safety': 0, 'mobility': 0, 'pawn_structure': 0}
        }
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0, 'D': 15, 'C': 2}
        center_hexes = [(0, 0), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]

        for q, r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(q, r)
            if piece:
                color = piece[0]
                piece_type = piece[1]
                # Material balance
                stats[color]['material_balance'] += piece_values[piece_type]
                # Center control
                if (q, r) in center_hexes:
                    stats[color]['center_control'] += 1
                # Mobility
                moves = self.game.get_possible_moves(q, r)
                stats[color]['mobility'] += len(moves)
                # Pawn structure
                if piece_type == 'P':
                    if self.is_passed_pawn(q, r, color):
                        stats[color]['pawn_structure'] += 2
                    if self.is_connected_pawn(q, r, color):
                        stats[color]['pawn_structure'] += 1

        # King safety
        for color in ['w', 'b']:
            king_pos = self.find_king(color)
            if king_pos:
                stats[color]['king_safety'] = self.calculate_king_safety(*king_pos, color)

        # Normalize values
        for color in ['w', 'b']:
            stats[color]['center_control'] /= len(center_hexes)
            stats[color]['king_safety'] /= 6  # Maximum possible value
            stats[color]['mobility'] /= 100  # Arbitrary normalization
            stats[color]['pawn_structure'] /= 20  # Arbitrary normalization

        return stats

    def is_passed_pawn(self, q: int, r: int, color: str) -> bool:
        direction = 1 if color == 'w' else -1
        for dq in [-1, 0, 1]:
            for dr in range(1, self.game.board.size + 1):
                check_q = q + dq
                check_r = r + dr * direction
                if self.game.board.is_valid_cell(check_q, check_r):
                    piece = self.game.board.get_cell(check_q, check_r)
                    if piece and piece[0] != color and piece[1] == 'P':
                        return False
        return True

    def is_connected_pawn(self, q: int, r: int, color: str) -> bool:
        for dq, dr in [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]:
            check_q, check_r = q + dq, r + dr
            if self.game.board.is_valid_cell(check_q, check_r):
                piece = self.game.board.get_cell(check_q, check_r)
                if piece and piece[0] == color and piece[1] == 'P':
                    return True
        return False

    def find_king(self, color: str) -> Optional[Tuple[int, int]]:
        for q, r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(q, r)
            if piece == f'{color}K':
                return (q, r)
        return None

    def calculate_king_safety(self, king_q: int, king_r: int, color: str) -> float:
        safety = 0
        for dq, dr in [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]:
            check_q, check_r = king_q + dq, king_r + dr
            if self.game.board.is_valid_cell(check_q, check_r):
                piece = self.game.board.get_cell(check_q, check_r)
                if piece and piece[0] == color:
                    safety += 1
        return safety

    def _get_all_moves(self, color: str) -> List[Tuple[int, int, int, int]]:
        moves = []
        for q, r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(q, r)
            if piece and piece[0] == color:
                moves.extend([(q, r, to_q, to_r) for to_q, to_r in self.game.get_possible_moves(q, r)])
        return moves

    def save_ai_progress(self):
        if isinstance(self.white_player, BasicAI):
            self.white_player.save_learning_progress()
        if isinstance(self.black_player, BasicAI):
            self.black_player.save_learning_progress()

    def closeEvent(self, event):
        self.save_ai_progress()
        logger.info("AI progress saved on application close")
        super().closeEvent(event)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
