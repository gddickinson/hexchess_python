import random
from typing import List, Tuple, Dict, Any, Optional
import json
import os
import numpy as np
from hexchess_game import HexChess
from collections import deque, Counter
import copy
import logging
import time

logger = logging.getLogger('hexchess.ai')

class LearningModule:
    def __init__(self, initial_params: Dict[str, float] = None):
        self.params = initial_params or {
            'piece_values': {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100, 'D': 15, 'C': 2},  # Added 'C' for Prince
            'aggression': 0.5,
            'center_control': 0.6,
            'king_safety': 0.7,
            'pawn_structure': 0.5,
            'mobility': 0.6,
            'defensive': 0.5,
            'piece_protection': 0.5,
            'pawn_promotion': 0.7,
            'material_advantage': 0.8,
            'piece_development': 0.6,
            'board_control': 0.7,
            'piece_coordination': 0.6,
            'tempo': 0.5,
            'endgame_preparedness': 0.5,
        }
        self.learning_rate = 0.1
        self.games_played = 0
        self.wins = 0
        self.learning_history = []

        # Add new attributes for loop prevention
        self.position_history = deque(maxlen=10)  # Store last 10 positions
        self.move_history = deque(maxlen=20)  # Store last 20 moves
        self.repetition_threshold = 2  # Number of repetitions before penalizing
        self.diversity_bonus = 0.1  # Bonus for diverse moves


    def update_params(self, result: float, game_stats: Dict[str, Any]):
        self.games_played += 1
        if result > 0.5:
            self.wins += 1

        for param, value in game_stats.items():
            if param in self.params:
                current_value = self.params[param]
                new_value = current_value + self.learning_rate * (value - current_value) * result
                self.params[param] = max(0, min(1, new_value))  # Clamp values between 0 and 1

        self.learning_history.append({
            'games_played': self.games_played,
            'win_rate': self.wins / self.games_played,
            'params': self.params.copy()
        })

    def _update_param(self, param: str, game_value: float, result: float) -> float:
        current_value = self.params[param]
        change = self.learning_rate * (game_value - current_value) * result
        change += random.uniform(-0.01, 0.01)
        new_value = current_value + change
        return max(0, min(1, new_value))  # Clamp values between 0 and 1


    def save_params(self, filename: str):
        try:
            data = {
                'params': self.params,
                'games_played': self.games_played,
                'wins': self.wins,
                'learning_history': self.learning_history
            }
            with open(filename, 'w') as f:
                json.dump(data, f)
            logger.info(f"AI parameters saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving AI parameters to {filename}: {str(e)}")

    def load_params(self, filename: str):
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                self.params = data['params']
                self.games_played = data['games_played']
                self.wins = data['wins']
                self.learning_history = data['learning_history']
                logger.info(f"AI parameters loaded from {filename}")
                return True
            else:
                logger.info(f"No existing parameter file found at {filename}")
                return False
        except Exception as e:
            logger.error(f"Error loading AI parameters from {filename}: {str(e)}")
            return False

    def get_save_data(self) -> Dict[str, Any]:
        return {
            'params': self.params,
            'games_played': self.games_played,
            'wins': self.wins,
            'learning_history': self.learning_history
        }

    def load_save_data(self, data: Dict[str, Any]) -> None:
        self.params = data['params']
        self.games_played = data['games_played']
        self.wins = data['wins']
        self.learning_history = data['learning_history']


class BasicAI:
    def __init__(self, game, color: str, custom_filename: str = None):
        self.game = game
        self.color = color
        self.learning_module = LearningModule()
        self.custom_filename = custom_filename
        self.logger = logging.getLogger('hexchess.ai')
        self.games_played = 0
        self.wins = 0
        self.total_moves = 0
        self.load_learning_progress()
        self.position_history = deque(maxlen=20)  # Increase history size
        self.move_history = deque(maxlen=40)  # Increase move history size
        self.repetition_threshold = 2  # Number of repetitions before penalizing
        self.diversity_bonus = 0.2  # Increase diversity bonus
        self.long_loop_threshold = 10  # Threshold for detecting longer loops
        self.forced_random_move_threshold = 30  # Force a random move after this many moves without progress
        self.performance_history = []  # Add this line to store performance history
        self.king_move_count = 0
        self.early_game_threshold = 10  # Consider first 10 moves as early game



        self.move_stats = {
            'aggression': 0,
            'center_control': 0,
            'king_safety': 0,
            'pawn_structure': 0,
            'mobility': 0,
            'defensive': 0,
            'piece_protection': 0,
            'pawn_promotion': 0,
            'material_advantage': 0,
            'piece_effectiveness': {piece: 0 for piece in 'PNBRQKD'},
            # New move stats
            'piece_development': 0,
            'board_control': 0,
            'piece_coordination': 0,
            'tempo': 0,
            'endgame_preparedness': 0,
        }

        # Define attack and defense values for each piece
        self.piece_values = {
            'P': {'attack': 1, 'defense': 1},
            'N': {'attack': 3, 'defense': 2},
            'B': {'attack': 3, 'defense': 2},
            'R': {'attack': 4, 'defense': 4},
            'Q': {'attack': 10, 'defense': 10},
            'K': {'attack': 2, 'defense': 5},
            'D': {'attack': 20, 'defense': 10},
            'C': {'attack': 2, 'defense': 2}
        }


    def penalize_parameters(self, penalty_factor: float):
        """
        Penalize the AI's parameters when a game is exited due to being stuck in a loop.
        """
        logger.info(f"{self.color} AI: Penalizing parameters with factor {penalty_factor}")
        for param in self.learning_module.params:
            if isinstance(self.learning_module.params[param], (int, float)):
                self.learning_module.params[param] *= penalty_factor
            elif isinstance(self.learning_module.params[param], dict):
                for sub_param in self.learning_module.params[param]:
                    self.learning_module.params[param][sub_param] *= penalty_factor

        # Penalize the diversity bonus and increase the repetition penalty
        self.diversity_bonus *= penalty_factor
        self.repetition_threshold = max(1, self.repetition_threshold - 1)

        # Log the penalized parameters
        logger.info(f"Penalized parameters: {self.learning_module.params}")
        logger.info(f"New diversity bonus: {self.diversity_bonus}")
        logger.info(f"New repetition threshold: {self.repetition_threshold}")



    def make_move(self) -> Tuple[int, int, int, int]:
        all_moves = self._get_all_possible_moves()
        if not all_moves:
            self.logging.warning(f"{self.color} AI: No possible moves found")
            return None


        # Check for immediate king captures
        king_capture_moves = self._find_king_capture_moves(all_moves)
        if king_capture_moves:
            chosen_move = random.choice(king_capture_moves)
            self.logger.info(f"{self.color} AI: Capturing king with move {chosen_move}")
            return chosen_move


        game_phase = self._determine_game_phase()
        move_scores = [(move, self._evaluate_move(move, game_phase)) for move in all_moves]

        # Apply diversity bonus
        move_scores = self._apply_diversity_bonus(move_scores)

        # Check for long loops and force a random move if necessary
        if self._detect_long_loop():
            logging.info(f"{self.color} AI: Long loop detected, forcing a random move")
            return random.choice(all_moves)

        best_move, best_score = max(move_scores, key=lambda x: x[1])

        # Update histories and counters
        self._update_histories(best_move)

        logging.info(f"{self.color} AI: Selected move {best_move} with score {best_score}")
        logging.debug(f"{self.color} AI: All move scores: {move_scores}")

        return best_move

    def _find_king_capture_moves(self, moves: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        king_captures = []
        for move in moves:
            from_q, from_r, to_q, to_r = move
            target_piece = self.game.board.get_cell(to_q, to_r)
            if target_piece and target_piece[1] == 'K' and target_piece[0] != self.color:
                king_captures.append(move)
        return king_captures

    def _determine_game_phase(self) -> str:
        total_pieces = sum(1 for _ in self.game.board.get_all_cells() if self.game.board.get_cell(*_))
        if total_pieces >= 24:  # You can adjust these thresholds
            return 'opening'
        elif total_pieces >= 12:
            return 'middlegame'
        else:
            return 'endgame'



    def _get_all_possible_moves(self) -> List[Tuple[int, int, int, int]]:
        all_moves = []
        for q, r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(q, r)
            if piece and piece[0] == self.color:
                moves = self.game.get_possible_moves(q, r)
                all_moves.extend([(q, r, move[0], move[1]) for move in moves])

        logging.debug(f"{self.color} AI: All possible moves: {all_moves}")
        return all_moves

    def _evaluate_move(self, move: Tuple[int, int, int, int], game_phase: str) -> float:
        from_q, from_r, to_q, to_r = move
        score = 0

        moving_piece = self.game.board.get_cell(from_q, from_r)
        captured_piece = self.game.board.get_cell(to_q, to_r)

        # Prioritize king captures
        if captured_piece and captured_piece[1] == 'K':
            return float(100000)  # Return highest possible score for king capture


        # Material value and piece-specific evaluations
        if captured_piece:
            score += self._evaluate_capture(moving_piece, captured_piece)

        # Evaluate based on game phase
        score += self._evaluate_phase_specific(moving_piece, from_q, from_r, to_q, to_r, game_phase)

        # Penalize early king moves
        if moving_piece[1] == 'K' and self.total_moves < self.early_game_threshold:
            score -= 50  # Substantial penalty for moving the king early

        # Evaluate other factors
        score += self._evaluate_position(to_q, to_r)
        score += self._evaluate_king_safety(to_q, to_r)
        score += self._evaluate_mobility(to_q, to_r)
        score += self._evaluate_piece_coordination(to_q, to_r)
        score += self._evaluate_tempo(moving_piece, from_q, from_r, to_q, to_r)


        # Add evaluation for Prince
        if moving_piece[1] == 'C':
            score += self._evaluate_prince_move(from_q, from_r, to_q, to_r)


        # Add stronger repetition penalty
        score += self._evaluate_repetition(move) * 2  # Double the repetition penalty

        # Add progress encouragement
        score += self._evaluate_progress(move)

        # Add stalemate avoidance
        score += self._evaluate_stalemate_avoidance(move)

        # Encourage development of other pieces in early game
        if self.total_moves < self.early_game_threshold:
            score += self._evaluate_early_game_development(moving_piece, to_q, to_r)

        return score


    def _evaluate_prince_move(self, from_q: int, from_r: int, to_q: int, to_r: int) -> float:
        score = 0
        # Encourage Prince to move towards the center
        center_distance = max(abs(to_q), abs(to_r))
        score += (self.game.board.size - center_distance) * 0.1

        # Encourage Prince to protect other pieces
        for dq, dr in [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]:
            adj_q, adj_r = to_q + dq, to_r + dr
            if self.game.board.is_valid_cell(adj_q, adj_r):
                adj_piece = self.game.board.get_cell(adj_q, adj_r)
                if adj_piece and adj_piece[0] == self.color:
                    score += 0.5

        return score

    def _evaluate_early_game_development(self, piece: str, to_q: int, to_r: int) -> float:
        score = 0
        if piece[1] in ['N', 'B', 'R']:
            # Encourage moving towards the center
            distance_to_center = max(abs(to_q), abs(to_r))
            score += (self.game.board.size - distance_to_center) * 0.5

            # Bonus for moving out of the back rank
            if (piece[0] == 'w' and to_r > -self.game.board.size) or \
               (piece[0] == 'b' and to_r < self.game.board.size):
                score += 2
        return score

    def _evaluate_king_safety(self, q: int, r: int) -> float:
        king_pos = self._find_king()
        if king_pos:
            king_q, king_r = king_pos
            distance_to_king = self.game.board.distance(q, r, king_q, king_r)

            # In early game, prefer pieces staying close to the king
            if self.total_moves < self.early_game_threshold:
                return self.learning_module.params['king_safety'] * (1 / (distance_to_king + 1))
            else:
                # In mid to late game, allow more flexibility
                return self.learning_module.params['king_safety'] * (1 / (distance_to_king + 1)) * 0.5
        return 0

    def _evaluate_repetition(self, move: Tuple[int, int, int, int]) -> float:
        from_q, from_r, to_q, to_r = move
        current_position = self._get_board_state_after_move(from_q, from_r, to_q, to_r)

        # Count occurrences of the resulting position
        repetitions = self.position_history.count(current_position)

        # Apply a stronger penalty if the position has been repeated
        if repetitions >= self.repetition_threshold:
            return -2.0 * repetitions  # Increase penalty for more repetitions
        return 0


    def _evaluate_progress(self, move: Tuple[int, int, int, int]) -> float:
        from_q, from_r, to_q, to_r = move

        # Encourage moves that haven't been made recently
        if move in self.move_history:
            return -0.5 * (len(self.move_history) - self.move_history.index(move))

        # Encourage moves towards the opponent's side
        progress = to_r - from_r if self.color == 'w' else from_r - to_r
        return progress * 0.1

    def _detect_long_loop(self) -> bool:
        if len(self.position_history) < self.long_loop_threshold:
            return False

        recent_positions = list(self.position_history)[-self.long_loop_threshold:]
        position_counts = Counter(recent_positions)

        # If any position appears more than half the time in recent history, consider it a long loop
        return any(count > self.long_loop_threshold // 2 for count in position_counts.values())



    def _evaluate_stalemate_avoidance(self, move: Tuple[int, int, int, int]) -> float:
        from_q, from_r, to_q, to_r = move

        # Create a hypothetical board state after the move
        hypothetical_board = self._get_hypothetical_board(from_q, from_r, to_q, to_r)

        # Check if the opponent has any legal moves in the hypothetical position
        opponent_color = 'b' if self.color == 'w' else 'w'
        opponent_has_moves = any(self._get_possible_moves_for_piece(q, r, hypothetical_board)
                                 for q, r in self.game.board.get_all_cells()
                                 if hypothetical_board.get_cell(q, r)
                                 and hypothetical_board.get_cell(q, r)[0] == opponent_color)

        # Return a bonus if the opponent has moves (avoiding stalemate)
        return 0.5 if opponent_has_moves else -1.0

    def _get_board_state_after_move(self, from_q: int, from_r: int, to_q: int, to_r: int) -> str:
        # Create a copy of the current board state
        board_state = self._get_board_state()

        # Apply the move to the copied state
        piece = self.game.board.get_cell(from_q, from_r)
        board_state = board_state[:self._get_index(from_q, from_r)] + '.' + board_state[self._get_index(from_q, from_r)+1:]
        board_state = board_state[:self._get_index(to_q, to_r)] + piece + board_state[self._get_index(to_q, to_r)+1:]

        return board_state

    def _get_hypothetical_board(self, from_q: int, from_r: int, to_q: int, to_r: int):
        # Create a copy of the current board
        hypothetical_board = self.game.board.__class__(self.game.board.size)
        for q, r in self.game.board.get_all_cells():
            hypothetical_board.set_cell(q, r, self.game.board.get_cell(q, r))

        # Apply the move to the hypothetical board
        piece = hypothetical_board.get_cell(from_q, from_r)
        hypothetical_board.set_cell(from_q, from_r, None)
        hypothetical_board.set_cell(to_q, to_r, piece)

        return hypothetical_board

    def _get_possible_moves_for_piece(self, q: int, r: int, board):
        piece = board.get_cell(q, r)
        if not piece:
            return []

        # This is a simplified version. You might need to implement a more complex logic
        # based on your game rules and piece movement patterns.
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        possible_moves = []
        for dq, dr in directions:
            new_q, new_r = q + dq, r + dr
            if board.is_valid_cell(new_q, new_r) and (board.get_cell(new_q, new_r) is None or
                                                      board.get_cell(new_q, new_r)[0] != piece[0]):
                possible_moves.append((new_q, new_r))
        return possible_moves

    def _get_index(self, q: int, r: int) -> int:
        # Convert q, r coordinates to an index in the board state string
        return (q + self.game.board.size) * (2 * self.game.board.size + 1) + (r + self.game.board.size)

    def _get_board_state(self) -> str:
        return ''.join(self.game.board.get_cell(q, r) or '.'
                       for q, r in self.game.board.get_all_cells())

    def _update_histories(self, move: Tuple[int, int, int, int]):
        self.move_history.append(move)
        from_q, from_r, to_q, to_r = move
        new_position = self._get_board_state_after_move(from_q, from_r, to_q, to_r)
        self.position_history.append(new_position)

        # Force a random move if no progress is made for a long time
        if len(self.move_history) >= self.forced_random_move_threshold:
            unique_moves = set(self.move_history)
            if len(unique_moves) <= self.forced_random_move_threshold // 2:
                logging.info(f"{self.color} AI: Forced random move due to lack of progress")
                all_moves = self._get_all_possible_moves()
                random_move = random.choice(all_moves)
                self.move_history.clear()
                self.position_history.clear()
                self._update_histories(random_move)

    def _apply_diversity_bonus(self, move_scores: List[Tuple[Tuple[int, int, int, int], float]]) -> List[Tuple[Tuple[int, int, int, int], float]]:
        move_counts = Counter(self.move_history)
        return [(move, score + self.diversity_bonus / (move_counts[move] + 1))
                for move, score in move_scores]


    def _evaluate_capture(self, moving_piece: str, captured_piece: str) -> float:
        if captured_piece[0] != self.color:
            attack_value = self.piece_values[moving_piece[1]]['attack']
            target_value = self.learning_module.params['piece_values'][captured_piece[1]]
            return attack_value * target_value * self.learning_module.params['aggression']
        else:
            return -self.learning_module.params['piece_values'][captured_piece[1]]

    def _evaluate_phase_specific(self, piece: str, from_q: int, from_r: int, to_q: int, to_r: int, phase: str) -> float:
        score = 0
        if phase == 'opening':
            score += self._evaluate_development(piece, from_q, from_r, to_q, to_r)
            score += self._evaluate_center_control(to_q, to_r)
        elif phase == 'middlegame':
            score += self._evaluate_piece_activity(piece, to_q, to_r)
            score += self._evaluate_pawn_structure(to_q, to_r)
        else:  # endgame
            score += self._evaluate_king_activity(piece, to_q, to_r)
            score += self._evaluate_pawn_promotion_potential(piece, to_q, to_r)
        return score

    def _evaluate_development(self, piece: str, from_q: int, from_r: int, to_q: int, to_r: int) -> float:
        if piece[1] in ['N', 'B'] and ((self.color == 'w' and from_r < 0) or (self.color == 'b' and from_r > 0)):
            return self.learning_module.params['piece_development']
        return 0

    def _evaluate_center_control(self, q: int, r: int) -> float:
        center_distance = max(abs(q), abs(r))
        return self.learning_module.params['center_control'] * (self.game.board.size - center_distance)

    def _evaluate_piece_activity(self, piece: str, q: int, r: int) -> float:
        moves = self.game.get_possible_moves(q, r)
        return len(moves) * self.learning_module.params['mobility'] * 0.1

    def _evaluate_pawn_structure(self, q: int, r: int) -> float:
        score = 0
        for dq, dr in [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]:
            if self.game.board.is_valid_cell(q + dq, r + dr):
                neighbor = self.game.board.get_cell(q + dq, r + dr)
                if neighbor and neighbor[0] == self.color and neighbor[1] == 'P':
                    score += self.learning_module.params['pawn_structure']
        return score

    def _evaluate_king_activity(self, piece: str, q: int, r: int) -> float:
        if piece[1] == 'K':
            center_distance = max(abs(q), abs(r))
            return self.learning_module.params['endgame_preparedness'] * (self.game.board.size - center_distance)
        return 0

    def _evaluate_pawn_promotion_potential(self, piece: str, q: int, r: int) -> float:
        if piece[1] == 'P':
            distance_to_promote = self.game.board.size - r if self.color == 'w' else r + self.game.board.size
            return self.learning_module.params['pawn_promotion'] * (self.game.board.size - distance_to_promote)
        return 0

    def _evaluate_position(self, q: int, r: int) -> float:
        return self.learning_module.params['board_control'] * (self.game.board.size - max(abs(q), abs(r)))


    def _evaluate_mobility(self, q: int, r: int) -> float:
        moves = self.game.get_possible_moves(q, r)
        return len(moves) * self.learning_module.params['mobility'] * 0.1

    def _evaluate_piece_coordination(self, q: int, r: int) -> float:
        score = 0
        for dq, dr in [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]:
            if self.game.board.is_valid_cell(q + dq, r + dr):
                neighbor = self.game.board.get_cell(q + dq, r + dr)
                if neighbor and neighbor[0] == self.color:
                    score += self.learning_module.params['piece_coordination']
        return score

    def _evaluate_tempo(self, piece: str, from_q: int, from_r: int, to_q: int, to_r: int) -> float:
        distance_moved = self.game.board.distance(from_q, from_r, to_q, to_r)
        return distance_moved * self.learning_module.params['tempo'] * 0.1

    def _find_king(self) -> Optional[Tuple[int, int]]:
        for q, r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(q, r)
            if piece == f'{self.color}K':
                return (q, r)
        return None


    def _is_piece_under_attack(self, q: int, r: int) -> bool:
        opponent_color = 'b' if self.color == 'w' else 'w'
        for opp_q, opp_r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(opp_q, opp_r)
            if piece and piece[0] == opponent_color:
                if (q, r) in self.game.get_possible_moves(opp_q, opp_r):
                    return True
        return False

    def _is_move_protective(self, from_q: int, from_r: int, to_q: int, to_r: int) -> bool:
        for q, r in self.game.board.get_adjacent_cells(to_q, to_r):
            piece = self.game.board.get_cell(q, r)
            if piece and piece[0] == self.color and self._is_piece_under_attack(q, r):
                return True
        return False


    def _is_pawn_near_promotion(self, q: int, r: int) -> bool:
        if self.color == 'w':
            return q + r >= self.game.board.size - 2
        else:
            return q + r <= -self.game.board.size + 2

    def _evaluate_material_advantage(self) -> float:
        material_score = 0
        for q, r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(q, r)
            if piece:
                value = self.learning_module.params['piece_values'][piece[1]]
                if piece[0] == self.color:
                    material_score += value
                else:
                    material_score -= value
        return material_score / 100  # Normalize the score


    def learn_from_game(self, result: float, game_stats: Dict[str, Any]):
        self.games_played += 1
        if result > 0.5:
            self.wins += 1

        # Normalize move stats
        total_moves = sum(self.move_stats['piece_effectiveness'].values())
        if total_moves > 0:
            for piece in self.move_stats['piece_effectiveness']:
                self.move_stats['piece_effectiveness'][piece] /= total_moves

            for param in ['aggression', 'center_control', 'king_safety', 'pawn_structure', 'mobility',
                          'defensive', 'piece_protection', 'pawn_promotion', 'material_advantage',
                          'piece_development', 'board_control', 'piece_coordination', 'tempo', 'endgame_preparedness']:
                if self.move_stats[param] > 0:
                    self.move_stats[param] /= total_moves

            # Add king move ratio to stats
            self.move_stats['king_move_ratio'] = self.king_move_count / total_moves

            # Adjust learning based on king move ratio
            if self.move_stats['king_move_ratio'] > 0.2:  # If king moved more than 20% of the time
                self.learning_module.params['king_safety'] *= 0.9  # Reduce importance of king safety

        # Combine AI's move stats with game stats
        combined_stats = {**game_stats, **self.move_stats}

        # Update learning module
        self.learning_module.update_params(result, combined_stats)

        # Add performance data to history
        self.performance_history.append({
            'game_number': self.games_played,
            'result': result,
            'material_balance': combined_stats.get('material_balance', 0),
            'center_control': combined_stats.get('center_control', 0),
            'king_safety': combined_stats.get('king_safety', 0),
            'mobility': combined_stats.get('mobility', 0),
            'king_move_ratio': combined_stats.get('king_move_ratio', 0),
            'piece_effectiveness': combined_stats.get('piece_effectiveness', {})
        })

        # Keep only the last 100 games in performance history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Reset move stats for the next game
        for key in self.move_stats:
            if isinstance(self.move_stats[key], dict):
                self.move_stats[key] = {k: 0 for k in self.move_stats[key]}
            else:
                self.move_stats[key] = 0

        self.king_move_count = 0
        self.total_moves = 0

        # Log learning progress
        logger.info(f"Game {self.games_played} completed. Result: {result}. Updated parameters: {self.learning_module.params}")



    def get_learning_stats(self) -> Dict[str, Any]:
        return {
            'games_played': self.games_played,
            'wins': self.wins,
            'win_rate': self.wins / self.games_played if self.games_played > 0 else 0,
            'avg_moves_per_game': self.total_moves / self.games_played if self.games_played > 0 else 0,
            'performance_history': self.performance_history,
            'current_parameters': self.learning_module.params
        }


    def get_filename(self):
        if self.custom_filename:
            return self.custom_filename
        return f'ai_params_{self.color}.json'

    def save_learning_progress(self):
        filename = self.get_filename()
        full_path = os.path.abspath(filename)
        print(f"Current working directory: {os.getcwd()}")
        print(f"Attempting to save AI progress to: {full_path}")
        try:
            with open(full_path, 'w') as f:
                json.dump(self.learning_module.get_save_data(), f)
            if os.path.exists(full_path):
                print(f"AI progress successfully saved to: {full_path}")
                print(f"File size: {os.path.getsize(full_path)} bytes")
                print(f"File permissions: {oct(os.stat(full_path).st_mode)[-3:]}")
            else:
                print(f"Failed to save AI progress. File not found at: {full_path}")
        except Exception as e:
            print(f"Error while saving AI progress: {str(e)}")

    def load_learning_progress(self):
        filename = self.get_filename()
        full_path = os.path.abspath(filename)
        print(f"Attempting to load AI progress from: {full_path}")
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            self.learning_module.load_save_data(data)
            print(f"AI progress successfully loaded from: {full_path}")
            return True
        except FileNotFoundError:
            print(f"Failed to load AI progress. File not found at: {full_path}")
            return False
        except Exception as e:
            print(f"Error while loading AI progress: {str(e)}")
            return False




class AdvancedAI(BasicAI):
    def __init__(self, game, color: str, custom_filename: str = None):
        super().__init__(game, color, custom_filename)
        self.max_time = 6  # Maximum time in seconds for a move
        self.max_depth = 5  # Maximum depth for iterative deepening
        self.transposition_table: Dict[str, Tuple[float, int, Tuple[int, int, int, int]]] = {}
        self.move_ordering: Dict[Tuple[int, int, int, int], float] = {}

    def make_move(self) -> Optional[Tuple[int, int, int, int]]:
        self.start_time = time.time()
        best_move = None
        best_score = float('-inf') if self.color == 'w' else float('inf')

        for depth in range(1, self.max_depth + 1):
            move, score = self._iterative_deepening(depth)
            if move is not None:
                best_move = move
                best_score = score
            if time.time() - self.start_time > self.max_time:
                break

        if best_move is None:
            # If no move was found (unlikely), fall back to a random move
            all_moves = self._get_all_possible_moves()
            if all_moves:
                best_move = random.choice(all_moves)

        return best_move

    def _iterative_deepening(self, depth: int) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        best_move = None
        best_score = float('-inf') if self.color == 'w' else float('inf')

        all_moves = self._get_all_possible_moves()
        all_moves.sort(key=lambda m: self.move_ordering.get(m, 0), reverse=True)

        for move in all_moves:
            if time.time() - self.start_time > self.max_time:
                break

            self.game.make_move(*move)
            score = self._minimax(depth - 1, float('-inf'), float('inf'), self.color != 'w')
            #self.game.undo_move()

            if (self.color == 'w' and score > best_score) or (self.color == 'b' and score < best_score):
                best_score = score
                best_move = move

            # Update move ordering
            self.move_ordering[move] = score

        return best_move, best_score

    def _minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
        if depth == 0 or self.game.is_game_over() or time.time() - self.start_time > self.max_time:
            return self._evaluate_board(self.game)

        board_hash = self._get_board_hash()
        if board_hash in self.transposition_table:
            stored_score, stored_depth, _ = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_score

        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = None

        all_moves = self._get_all_possible_moves()
        all_moves.sort(key=lambda m: self.move_ordering.get(m, 0), reverse=maximizing_player)

        for move in all_moves:
            self.game.make_move(*move)
            score = self._minimax(depth - 1, alpha, beta, not maximizing_player)
            self.game.undo_move()

            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)

            if beta <= alpha:
                break

        self.transposition_table[board_hash] = (best_score, depth, best_move)
        return best_score

    def _get_board_hash(self) -> str:
        return ''.join(self.game.board.get_cell(q, r) or '.' for q, r in self.game.board.get_all_cells())



    def _minimax_root(self, depth: int) -> Tuple[int, int, int, int]:
        best_move = None
        best_value = float('-inf') if self.color == 'w' else float('inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in self._get_all_possible_moves():
            game_copy = copy.deepcopy(self.game)
            game_copy.make_move(*move)
            value = self._minimax(game_copy, depth - 1, alpha, beta, self.color != 'w')

            if self.color == 'w':
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)

            if beta <= alpha:
                break

        return best_move

    def _get_all_possible_moves_for_game(self, game: HexChess) -> List[Tuple[int, int, int, int]]:
        all_moves = []
        for q, r in game.board.get_all_cells():
            piece = game.board.get_cell(q, r)
            if piece and piece[0] == game.current_player:
                moves = game.get_possible_moves(q, r)
                all_moves.extend([(q, r, move[0], move[1]) for move in moves])
        return all_moves

    def _evaluate_board(self, game: HexChess) -> float:
        if game.is_game_over():
            winner = game.get_winner()
            if winner == self.color:
                return float('inf')
            elif winner == 'draw':
                return 0
            else:
                return float('-inf')

        score = 0

        # Material balance
        score += self._evaluate_material_balance(game)

        # Piece-specific evaluations
        score += self._evaluate_piece_positions(game)
        score += self._evaluate_king_safety(game)
        score += self._evaluate_pawn_structure(game)

        # Control of the center
        score += self._evaluate_center_control(game)

        # Mobility
        score += self._evaluate_mobility(game)

        # Piece coordination
        score += self._evaluate_piece_coordination(game)

        return score if self.color == 'w' else -score


    def _evaluate_king_safety(self, game: HexChess) -> float:
        king_pos = self._find_king_position(game, self.color)
        if not king_pos:
            return 0

        safety_score = 0
        adjacent_hexes = game.board.get_adjacent_cells(*king_pos)

        # Count friendly pieces around the king
        for adj_q, adj_r in adjacent_hexes:
            piece = game.board.get_cell(adj_q, adj_r)
            if piece and piece[0] == self.color:
                safety_score += 1

        # Penalize if the king is in the center
        center_distance = max(abs(king_pos[0]), abs(king_pos[1]))
        safety_score += center_distance * 0.1

        return safety_score

    def _evaluate_pawn_structure(self, game: HexChess) -> float:
        score = 0
        for q, r in game.board.get_all_cells():
            piece = game.board.get_cell(q, r)
            if piece and piece[1] == 'P':
                if piece[0] == self.color:
                    score += self._evaluate_pawn_position(game, q, r, self.color)
                else:
                    score -= self._evaluate_pawn_position(game, q, r, self._opposite_color(self.color))
        return score

    def _evaluate_center_control(self, game: HexChess) -> float:
        center_hexes = [(0, 0), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]
        score = 0
        for q, r in center_hexes:
            piece = game.board.get_cell(q, r)
            if piece:
                value = 0.1 * self.piece_values[piece[1]]['attack']
                score += value if piece[0] == self.color else -value
        return score

    def _evaluate_mobility(self, game: HexChess) -> float:
        my_mobility = sum(len(game.get_possible_moves(q, r))
                          for q, r in game.board.get_all_cells()
                          if game.board.get_cell(q, r) and game.board.get_cell(q, r)[0] == self.color)

        opponent_mobility = sum(len(game.get_possible_moves(q, r))
                                for q, r in game.board.get_all_cells()
                                if game.board.get_cell(q, r) and game.board.get_cell(q, r)[0] != self.color)

        return (my_mobility - opponent_mobility) * 0.1

    def _evaluate_piece_coordination(self, game: HexChess) -> float:
        score = 0
        for q, r in game.board.get_all_cells():
            piece = game.board.get_cell(q, r)
            if piece and piece[0] == self.color:
                adjacent_hexes = game.board.get_adjacent_cells(q, r)
                for adj_q, adj_r in adjacent_hexes:
                    adj_piece = game.board.get_cell(adj_q, adj_r)
                    if adj_piece and adj_piece[0] == self.color:
                        score += 0.1
        return score

    def _evaluate_pawn_position(self, game: HexChess, q: int, r: int, color: str) -> float:
        score = 0
        direction = 1 if color == 'w' else -1

        # Bonus for advanced pawns
        score += r * direction * 0.1

        # Penalty for doubled pawns
        if game.board.get_cell(q, r + direction) == f"{color}P":
            score -= 0.5

        # Bonus for connected pawns
        adjacent_hexes = game.board.get_adjacent_cells(q, r)
        for adj_q, adj_r in adjacent_hexes:
            if game.board.get_cell(adj_q, adj_r) == f"{color}P":
                score += 0.3

        return score

    def _evaluate_minor_piece_position(self, game: HexChess, q: int, r: int, color: str) -> float:
        score = 0

        # Encourage development in the opening
        if len(game.move_history) < 10:
            if (color == 'w' and r > -4) or (color == 'b' and r < 4):
                score += 0.5

        # Control of center
        center_distance = max(abs(q), abs(r))
        score += (7 - center_distance) * 0.1

        return score

    def _evaluate_rook_position(self, game: HexChess, q: int, r: int, color: str) -> float:
        score = 0

        # Bonus for rooks on open files
        if not any(game.board.get_cell(q, r2) and game.board.get_cell(q, r2)[1] == 'P'
                   for r2 in range(-game.board.size, game.board.size + 1)):
            score += 0.5

        # Bonus for rooks on the 7th rank (or 2nd rank for black)
        if (color == 'w' and r == 7) or (color == 'b' and r == -7):
            score += 0.5

        return score

    def _evaluate_queen_position(self, game: HexChess, q: int, r: int, color: str) -> float:
        score = 0

        # Penalize early queen development
        if len(game.move_history) < 10:
            if (color == 'w' and r > -3) or (color == 'b' and r < 3):
                score -= 0.5

        # Bonus for central queen in the endgame
        if self._is_endgame(game):
            center_distance = max(abs(q), abs(r))
            score += (7 - center_distance) * 0.1

        return score

    def _evaluate_king_position(self, game: HexChess, q: int, r: int, color: str) -> float:
        score = 0

        if self._is_endgame(game):
            # In endgame, king should be active
            center_distance = max(abs(q), abs(r))
            score += (7 - center_distance) * 0.1
        else:
            # In midgame and opening, king should stay back
            back_rank = -7 if color == 'w' else 7
            score += abs(r - back_rank) * -0.1

        return score

    def _evaluate_dragon_position(self, game: HexChess, q: int, r: int, color: str) -> float:
        score = 0

        # Dragons are powerful, encourage central control
        center_distance = max(abs(q), abs(r))
        score += (7 - center_distance) * 0.2

        # Bonus for attacking potential
        adjacent_hexes = game.board.get_adjacent_cells(q, r)
        for adj_q, adj_r in adjacent_hexes:
            piece = game.board.get_cell(adj_q, adj_r)
            if piece and piece[0] != color:
                score += 0.1

        return score

    def _is_endgame(self, game: HexChess) -> bool:
        total_pieces = sum(1 for _ in game.board.get_all_cells() if game.board.get_cell(*_))
        return total_pieces <= 10  # Adjust this threshold as needed

    def _find_king_position(self, game: HexChess, color: str) -> Optional[Tuple[int, int]]:
        for q, r in game.board.get_all_cells():
            piece = game.board.get_cell(q, r)
            if piece and piece[1] == 'K' and piece[0] == color:
                return (q, r)
        return None

    def _opposite_color(self, color: str) -> str:
        return 'b' if color == 'w' else 'w'



    def _evaluate_material_balance(self, game: HexChess) -> float:
        balance = 0
        for q, r in game.board.get_all_cells():
            piece = game.board.get_cell(q, r)
            if piece:
                value = self.piece_values[piece[1]]['attack']
                balance += value if piece[0] == 'w' else -value
        return balance

    def _evaluate_piece_positions(self, game: HexChess) -> float:
        score = 0
        for q, r in game.board.get_all_cells():
            piece = game.board.get_cell(q, r)
            if piece:
                if piece[1] == 'P':
                    score += self._evaluate_pawn_position(game, q, r, piece[0])
                elif piece[1] in ['N', 'B']:
                    score += self._evaluate_minor_piece_position(game, q, r, piece[0])
                elif piece[1] == 'R':
                    score += self._evaluate_rook_position(game, q, r, piece[0])
                elif piece[1] == 'Q':
                    score += self._evaluate_queen_position(game, q, r, piece[0])
                elif piece[1] == 'K':
                    score += self._evaluate_king_position(game, q, r, piece[0])
                elif piece[1] == 'D':
                    score += self._evaluate_dragon_position(game, q, r, piece[0])
        return score

    def _evaluate_piece_synergy(self) -> float:
        score = 0
        for q, r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(q, r)
            if piece and piece[0] == self.color:
                adjacent_friendly_pieces = sum(1 for adj_q, adj_r in self.game.board.get_adjacent_cells(q, r)
                                               if self.game.board.get_cell(adj_q, adj_r)
                                               and self.game.board.get_cell(adj_q, adj_r)[0] == self.color)
                score += adjacent_friendly_pieces * 0.1
        return score

    def _evaluate_king_tropism(self) -> float:
        opponent_color = 'b' if self.color == 'w' else 'w'
        opponent_king_pos = self._find_king(opponent_color)
        if not opponent_king_pos:
            return 0

        score = 0
        for q, r in self.game.board.get_all_cells():
            piece = self.game.board.get_cell(q, r)
            if piece and piece[0] == self.color:
                distance_to_enemy_king = self.game.board.distance(q, r, *opponent_king_pos)
                score += (14 - distance_to_enemy_king) * 0.05  # 14 is max possible distance on 8x8 board
        return score



    def _evaluate_single_pawn(self, q: int, r: int, direction: int) -> float:
        score = 0
        # Bonus for advanced pawns
        score += r * direction * 0.1
        # Penalty for doubled pawns
        if self.game.board.get_cell(q, r + direction) == f"{self.color}P":
            score -= 0.5
        # Bonus for connected pawns
        for dq, dr in [(1, 0), (-1, 0)]:
            if self.game.board.get_cell(q + dq, r + dr) == f"{self.color}P":
                score += 0.3
        return score



    def _evaluate_endgame(self) -> float:
        score = 0
        my_king_pos = self._find_king(self.color)
        opponent_king_pos = self._find_king('b' if self.color == 'w' else 'w')

        if my_king_pos and opponent_king_pos:
            # Drive opponent's king to the edge
            opponent_king_center_distance = max(abs(opponent_king_pos[0]), abs(opponent_king_pos[1]))
            score += opponent_king_center_distance * 0.1

            # Bring our king to the center
            my_king_center_distance = max(abs(my_king_pos[0]), abs(my_king_pos[1]))
            score -= my_king_center_distance * 0.1

            # Reduce distance between kings
            king_distance = self.game.board.distance(*my_king_pos, *opponent_king_pos)
            score -= king_distance * 0.05

        return score

    def _initialize_opening_book(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        # This is a simple opening book. You can expand it with more sophisticated openings.
        return {
            self._get_initial_board_state(): [
                (-8, 0, -7, 0),  # Move rook
                (-7, -1, -5, -2),  # Move knight
                (-6, -2, -4, -4),  # Move bishop
            ]
        }

    def _get_initial_board_state(self) -> str:
        initial_game = HexChess()
        return self._get_board_state_from_game(initial_game)

    def _get_board_state(self) -> str:
        return self._get_board_state_from_game(self.game)

    @staticmethod
    def _get_board_state_from_game(game: HexChess) -> str:
        return ''.join(game.board.get_cell(q, r) or '.'
                       for q, r in game.board.get_all_cells())

    def learn_from_game(self, result: float, game_stats: Dict[str, Any]):
        super().learn_from_game(result, game_stats)

        # Additional learning for AdvancedAI
        self._update_opening_book(result)
        self._adjust_endgame_threshold(game_stats)

    def _update_opening_book(self, result: float):
        if len(self.game.move_history) > 10:  # Only consider games that went beyond the opening
            opening_sequence = tuple(self.game.move_history[:5])  # Consider first 5 moves as the opening
            if opening_sequence in self.opening_book:
                self.opening_book[opening_sequence] += result - 0.5  # Adjust the score
            else:
                self.opening_book[opening_sequence] = result - 0.5

    def _adjust_endgame_threshold(self, game_stats: Dict[str, Any]):
        if 'total_pieces' in game_stats:
            # Slowly adjust the endgame threshold based on game results
            adjustment = 0.1 if game_stats['total_pieces'] < self.endgame_threshold else -0.1
            self.endgame_threshold += adjustment
            self.endgame_threshold = max(5, min(20, self.endgame_threshold))  # Keep threshold between 5 and 20



