import math
import numpy as np

from enum import Enum
from functools import reduce

from negamax import negamax

class Color(Enum):
    WHITE = 1
    BLACK = 2

class Piece(Enum):
    PAWN = 1
    KING = 2

class Status(Enum):
    ONGOING = 1,
    DRAW = 2,
    WHITE_WIN = 3,
    BLACK_WIN = 4

class Opponent(Enum):
    RANDOM = 1,
    GREEDY = 2,
    OPTIMAL = 3

class Checkers:
    def __init__(self, shape=(8,8), opponent='random', absolute=False):
        self.width, self.height = shape

        self.squareToPosition = {
            1:  (0,1), 2:  (0,3), 3:  (0,5), 4:  (0,7),
            5:  (1,0), 6:  (1,2), 7:  (1,4), 8:  (1,6),
            9:  (2,1), 10: (2,3), 11: (2,5), 12: (2,7),
            13: (3,0), 14: (3,2), 15: (3,4), 16: (3,6),
            17: (4,1), 18: (4,3), 19: (4,5), 20: (4,7),
            21: (5,0), 22: (5,2), 23: (5,4), 24: (5,6),
            25: (6,1), 26: (6,3), 27: (6,5), 28: (6,7),
            29: (7,0), 30: (7,2), 31: (7,4), 32: (7,6),
        }

        self.positionToSquare = {
            (0,1): 1,  (0,3): 2,  (0,5): 3,  (0,7): 4,
            (1,0): 5,  (1,2): 6,  (1,4): 7,  (1,6): 8,
            (2,1): 9,  (2,3): 10, (2,5): 11, (2,7): 12,
            (3,0): 13, (3,2): 14, (3,4): 15, (3,6): 16,
            (4,1): 17, (4,3): 18, (4,5): 19, (4,7): 20,
            (5,0): 21, (5,2): 22, (5,4): 23, (5,6): 24,
            (6,1): 25, (6,3): 26, (6,5): 27, (6,7): 28,
            (7,0): 29, (7,2): 30, (7,4): 31, (7,6): 32
        }

        self.opponent = opponent
        self.absolute = absolute

    def _check_boundaries(self, position):
        x, y = position
        return x >= 0 and x < self.height and y >= 0 and y < self.width

    def _get_square_color(self, square):
        if square in self._state[Color.WHITE]:
            return Color.WHITE

        if square in self._state[Color.BLACK]:
            return Color.BLACK

        return None

    def _get_jump_square(self, source, target):
        source_x, source_y = self.squareToPosition[source]
        target_x, target_y = self.squareToPosition[target]

        # Determine the resulting position
        jump = (target_x + (target_x - source_x), target_y + (target_y - source_y))

        return self.positionToSquare[jump] if self._check_boundaries(jump) else None

    def _get_adjacents(self, color, piece, square):
        x, y = self.squareToPosition[square]

        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        if piece is Piece.PAWN:
            directions = directions[:2] if color is Color.WHITE else directions[2:]
        
        return [
            self.positionToSquare[(x + dir_x, y + dir_y)] for dir_x, dir_y in directions 
                if self._check_boundaries((x + dir_x, y + dir_y))
        ]

    def _get_available_actions(self, color, piece, square, initial=None, jumped=[]):
        moves = []
        jumps = []

        # Iterate over adjacent squares, depending on the color and piece we're moving
        for adj_square in self._get_adjacents(color, piece, square):
            if adj_square in jumped:
                continue

            adj_color = self._get_square_color(adj_square)

            # Single move
            if adj_color is None:
                moves.append(adj_square)
            
            # Jump move
            elif adj_color is not color:

                # Jump square, resulting from jumping over the adjacent square
                jump_square = self._get_jump_square(square, adj_square)
                jump_color = self._get_square_color(jump_square)

                # Check if jump square is free (or if it's the initial square)
                if jump_square is not None and (jump_square is initial or jump_color is None):
                    new_jumped = jumped + [adj_square]

                    # Recursive call using the resulting square (only considers jump moves)
                    _, j = self._get_available_actions(
                        color, piece, jump_square, initial=square if initial is None else initial, jumped=new_jumped
                    )
                    
                    # Jump square is the final jump, save jumped pieces
                    if len(j) == 0:
                        jumps.append((jump_square, new_jumped))
                    else:
                        jumps += j

        return moves, jumps

    def _should_promote(self, square):
        x, _ = self.squareToPosition[square]
        return (self.turn is Color.WHITE and x == 0) or (self.turn is Color.BLACK and x == self.height - 1)

    def reset(self, episode=1):
        self.episode = episode

        self._status = Status.ONGOING
        self._last_jumped = 0

        self.turn = Color.WHITE

        if self.width == 8:
            self._state = {
                Color.BLACK: {
                    1: Piece.PAWN, 2: Piece.PAWN, 3: Piece.PAWN, 4: Piece.PAWN,
                    5: Piece.PAWN, 6: Piece.PAWN,  7: Piece.PAWN, 8: Piece.PAWN,
                    9: Piece.PAWN, 10: Piece.PAWN,  11: Piece.PAWN, 12: Piece.PAWN
                },
                Color.WHITE: {
                    21: Piece.PAWN, 22: Piece.PAWN,  23: Piece.PAWN, 24: Piece.PAWN,
                    25: Piece.PAWN, 26: Piece.PAWN,  27: Piece.PAWN, 28: Piece.PAWN,
                    29: Piece.PAWN, 30: Piece.PAWN,  31: Piece.PAWN, 32: Piece.PAWN,
                }
            }
        elif self.width == 6:
            self._state = {
                Color.BLACK: {
                    1: Piece.PAWN, 2: Piece.PAWN, 3: Piece.PAWN, 
                    5: Piece.PAWN, 6: Piece.PAWN,  7: Piece.PAWN
                },
                Color.WHITE: {
                    17: Piece.PAWN, 18: Piece.PAWN, 19: Piece.PAWN,
                    21: Piece.PAWN, 22: Piece.PAWN,  23: Piece.PAWN,
                }
            }

    def state(self):
        if self.is_over:
            print('Game is already over')
            raise

        return self.feature_vector()

    def actions(self, color=None, only_legal=True):
        if self.is_over:
            print('Game is already over')
            raise

        color = self.turn if color is None else color

        all_moves = []
        all_jumps = []

        for square, piece in self._state[color].items():
            moves, jumps = self._get_available_actions(color, piece, square)

            for move in moves:
                all_moves.append((square, move, []))

            for jump, jumped in jumps:
                all_jumps.append((square, jump, jumped))

        if only_legal:
            return all_moves if len(all_jumps) == 0 else all_jumps
        else:
            return all_moves + all_jumps

    def step(self, action):
        last_evaluation = self.heuristic()
        
        self._step(action)

        if not self.is_over:
            self._step_opponent()

        curr_evaluation = self.heuristic()

        if self.is_over:
            if self._status is Status.WHITE_WIN:
                reward = 2.0
            elif self._status is Status.BLACK_WIN:
                reward = -2.0
            else:
                reward = 0.0
        else:
            reward = 0.0

        return None if self.is_over else self.state(), None if self.is_over else self.actions(), (reward, last_evaluation, curr_evaluation)

    def _step(self, action):
        if self.is_over:
            print('Game is already over')
            raise

        opponent = Color.BLACK if self.turn is Color.WHITE else Color.WHITE

        source, target, jumped = action
        piece = self._state[self.turn][source]

        # Move piece
        del self._state[self.turn][source]
        self._state[self.turn][target] = piece

        # Delete jumped pieces
        for j in jumped:
            del self._state[opponent][j]

        # Promote piece
        if self._should_promote(target):
            self._state[self.turn][target] = Piece.KING

        # Check winner
        if len(self._state[Color.BLACK]) == 0 or len(self.actions(Color.BLACK)) == 0:
            self._status = Status.WHITE_WIN
        elif len(self._state[Color.WHITE]) == 0 or len(self.actions(Color.WHITE)) == 0:
            self._status = Status.BLACK_WIN

        # Check last jump
        if len(jumped) == 0:
            self._last_jumped += 1
        else:
            self._last_jumped = 0
        
        if self._last_jumped >= 20:
            self._status = Status.DRAW

        self.turn = opponent

    def _step_opponent(self):
        state = self.state()
        actions = self.actions()
        
        if self.opponent == 'random':
            action = actions[np.random.choice(len(actions))]
        elif self.opponent == 'greedy':
            action = negamax(self, 1, -math.inf, math.inf, -1)[0]
        elif self.opponent == 'optimal3':
            action = negamax(self, 3, -math.inf, math.inf, -1)[0]
        elif self.opponent == 'optimal5':
            action = negamax(self, 5, -math.inf, math.inf, -1)[0]
        elif self.opponent == 'optimal7':
            action = negamax(self, 7, -math.inf, math.inf, -1)[0]
        elif self.opponent == 'greedy_to_optimal3':
            if self.width == 6:
                cutoff = 20000 
            elif self.width == 8:
                cutoff = 25000

            if self.episode < cutoff:
                action = negamax(self, 1, -math.inf, math.inf, -1)[0]
            else:
                action = negamax(self, 3, -math.inf, math.inf, -1)[0]

        self._step(action)
    
    def feature_vector(self):
        pawn_value = 1.0
        king_value = 2.0

        white_pieces = self._state[Color.WHITE]
        black_pieces = self._state[Color.BLACK]

        # Squares
        black_backrow_squares = range(1, 4) if self.width == 6 else range(1, 5)
        black_back_squares = range(5, 8) if self.width == 6 else range(5, 13)
        center_squares = [9, 10, 11, 13, 14, 15] if self.width == 6 else range(13, 21)
        white_back_squares = range(17, 20) if self.width == 6 else range(21, 29)
        white_backrow_squares = range(21, 24) if self.width == 6 else range(29, 33)
        left_squares = [1, 5, 6, 9, 13, 14, 17, 21, 22] if self.width == 6 else [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30]
        right_squares = [2, 3, 7, 10, 11, 15, 18, 19, 23] if self.width == 6 else [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31, 32]
         
        # Piece count
        black_backrow = [0.0, 0.0]
        black_back = [0.0, 0.0]
        center = [0.0, 0.0]
        white_back = [0.0, 0.0]
        white_backrow = [0.0, 0.0]
        left = [0.0, 0.0]
        right = [0.0, 0.0]

        for i in range(2):
            pieces = white_pieces if i == 0 else black_pieces

            for square, piece in pieces.items():
                if square in black_backrow_squares:
                    black_backrow[i] += pawn_value if piece is Piece.PAWN else king_value
                elif square in black_back_squares:
                    black_back[i] += pawn_value if piece is Piece.PAWN else king_value
                elif square in center_squares:
                    center[i] += pawn_value if piece is Piece.PAWN else king_value
                elif square in white_back_squares:
                    white_back[i] += pawn_value if piece is Piece.PAWN else king_value
                elif square in white_backrow_squares:
                    white_backrow[i] += pawn_value if piece is Piece.PAWN else king_value

                if square in left_squares:
                    left[i] += pawn_value if piece is Piece.PAWN else king_value
                elif square in right_squares:
                    right[i] += pawn_value if piece is Piece.PAWN else king_value

        # Piece total count
        black_backrow_sum = np.sum(black_backrow)
        black_back_sum = np.sum(black_back)
        center_sum = np.sum(center)
        white_back_sum = np.sum(white_back)
        white_backrow_sum = np.sum(white_backrow)
        left_sum = np.sum(left)
        right_sum = np.sum(right)
        
        feature_vector = (
            black_backrow[0] - black_backrow[1],
            black_back[0] - black_back[1],
            center[0] - center[1],
            white_back[0] - white_back[1],
            white_backrow[0] - white_backrow[1],
            left[0] - left[1],
            right[0] - right[1],
            len(self.actions())
        ) if self.absolute else (
            (black_backrow[0] - black_backrow[1]) / black_backrow_sum if black_backrow_sum != 0.0 else 0.0,
            (black_back[0] - black_back[1]) / black_back_sum if black_back_sum != 0.0 else 0.0,
            (center[0] - center[1]) / center_sum if center_sum != 0.0 else 0.0,
            (white_back[0] - white_back[1]) / white_back_sum if white_back_sum != 0.0 else 0.0,
            (white_backrow[0] - white_backrow[1]) / white_backrow_sum if white_backrow_sum != 0.0 else 0.0,
            (left[0] - left[1]) / left_sum if left_sum != 0.0 else 0.0,
            (right[0] - right[1]) / right_sum if right_sum != 0.0 else 0.0,
            len(self.actions())
        )

        return feature_vector

    def heuristic(self):
        white_pieces = self._state[Color.WHITE]
        black_pieces = self._state[Color.BLACK]

        white_material_score = reduce(lambda x, key: x + (1.0 if white_pieces[key] is Piece.PAWN else 2.0), white_pieces, 0.0)
        black_material_score = reduce(lambda x, key: x + (1.0 if black_pieces[key] is Piece.PAWN else 2.0), black_pieces, 0.0)
        
        return (white_material_score - black_material_score) / (white_material_score + black_material_score)

    def evaluate(self):
        white_pieces = self._state[Color.WHITE]
        black_pieces = self._state[Color.BLACK]

        # Win
        win_weight = 50.0
        if self.is_over:
            return win_weight * (-1.0 if self._status is Status.DRAW else (1.0 if self._status is Status.WHITE_WIN else -1.0))

        # Material
        material_weight = 2.5
        material_score = material_weight * (reduce(
            lambda x, key: x + (1.0 if white_pieces[key] is Piece.PAWN else 2.0),
            white_pieces,
            0.0
        ) - reduce(
            lambda x, key: x + (1.0 if black_pieces[key] is Piece.PAWN else 2.0),
            black_pieces,
            0.0
        ))

        # Mobility
        # mobility_weight = 0.25
        # mobility_score = mobility_weight * (
        #     len(self.actions(color=Color.WHITE, only_legal=False)) - len(self.actions(color=Color.BLACK, only_legal=False))
        # )
        # mobility_score = 0.0

        # Threat
        threat_weight = 1.5 #0.5
        attacked_pieces = black_pieces if self.turn is Color.WHITE else white_pieces
        threat_score = threat_weight * (1 if self.turn is Color.WHITE else -1) * max(
            list(map(
                lambda j: reduce(lambda x, key: x + (1.0 if attacked_pieces[key] is Piece.PAWN else 2.0), j, 0.0),
                [j for _, _, j in self.actions(self.turn)]
            ))
        )

        # Pieces positioning
        position_weight = 1.0
        
        white_position = 0.0
        for square, piece in white_pieces.items():
            if piece is Piece.KING:
                white_position += 1.5
                continue
            
            x, y = self.squareToPosition[square]

            if self.width == 8:
                if x < 3:   # Front
                    white_position += 1.5
                elif x < 5: # Center
                    white_position += 1.4
                elif x < 7: # Back
                    white_position += 1.2
                else:       # Backrow
                    white_position += 1.0
            elif self.width == 6:
                if x < 2:
                    white_position += 1.5
                elif x < 4:
                    white_position += 1.4
                elif x < 5:
                    white_position += 1.2
                else:
                    white_position += 1.0
            
        black_position = 0.0
        for square, piece in black_pieces.items():
            if piece is Piece.KING:
                black_position += 1.5
                continue
        
            x, y = self.squareToPosition[square]

            if self.width == 8:
                if x > 4:   # Front
                    black_position += 1.5
                elif x > 2: # Center
                    black_position += 1.4
                elif x > 0: # Back
                    black_position += 1.2
                else:       # Backrow
                    black_position += 1.0
            elif self.width == 6:
                if x > 3:
                    black_position += 1.5
                elif x > 1:
                    black_position += 1.4
                elif x > 0:
                    black_position += 1.2
                else:
                    black_position += 1.0

        position_score = position_weight * (white_position - black_position)

        # Free king
        free_king_weight = 2.0

        white_free_king = 0.0
        for square, piece in white_pieces.items():
            if piece is Piece.KING:
                continue

            x, y = self.squareToPosition[square]

            if x >= 3:
                continue

            is_free = True

            for adj_x in range(x):
                if not is_free:
                    break

                for adj_y in range(y - (x - adj_x), y + (x - adj_x) + 1):
                    if adj_y < 0 or adj_y >= self.width or (adj_x, adj_y) not in self.positionToSquare:
                        continue

                    adj_square = self.positionToSquare[(adj_x, adj_y)]

                    if adj_square in white_pieces or adj_square in black_pieces:
                        is_free = False
                        break

            if is_free:
                white_free_king += 1

        for source, target, jumped in self.actions(Color.WHITE):
            if white_pieces[source] is Piece.KING or len(jumped) == 0:
                continue

            x, _ = self.squareToPosition[target]

            if x == 0:
                white_free_king += 1
        
        black_free_king = 0.0
        for square, piece in black_pieces.items():
            if piece is Piece.KING:
                continue

            x, y = self.squareToPosition[square]

            if x <= 4:
                continue

            is_free = True

            for adj_x in range(x + 1, self.height):
                if not is_free:
                    break

                for adj_y in range(y - (adj_x - x), y + (adj_x + x) + 1):
                    if adj_y < 0 or adj_y >= self.width or (adj_x, adj_y) not in self.positionToSquare:
                        continue

                    adj_square = self.positionToSquare[(adj_x, adj_y)]

                    if adj_square in white_pieces or adj_square in black_pieces:
                        is_free = False
                        break

            if is_free:
                black_free_king += 1

        for source, target, jumped in self.actions(Color.BLACK):
            if black_pieces[source] is Piece.KING or len(jumped) == 0:
                continue

            x, _ = self.squareToPosition[target]

            if x == self.height - 1:
                black_free_king += 1

        free_king_score = free_king_weight * (white_free_king - black_free_king)

        # print("EVAL:", material_score, threat_score, round(position_score, 3), free_king_score)
        
        return round(material_score + threat_score + position_score + free_king_score, 3)

    @property
    def is_over(self):
        return self._status is not Status.ONGOING

    @property
    def winner(self):
        if not self.is_over:
            print('Game not over yet')
            raise

        if self._status is Status.DRAW:
            return 'D'
        elif self._status is Status.WHITE_WIN:
            return 'W'
        elif self._status is Status.BLACK_WIN:
            return 'B'

    def display(self):
        if self.width == 8:
            board = np.array([
                ['--', '01', '--', '02', '--', '03', '--', '04'],
                ['05', '--', '06', '--', '07', '--', '08', '--'],
                ['--', '09', '--', '10', '--', '11', '--', '12'],
                ['13', '--', '14', '--', '15', '--', '16', '--'],
                ['--', '17', '--', '18', '--', '19', '--', '20'],
                ['21', '--', '22', '--', '23', '--', '24', '--'],
                ['--', '25', '--', '26', '--', '27', '--', '28'],
                ['29', '--', '30', '--', '31', '--', '32', '--']
            ])
        elif self.width == 6:
            board = np.array([
                ['--', '01', '--', '02', '--', '03'],
                ['05', '--', '06', '--', '07', '--'],
                ['--', '09', '--', '10', '--', '11'],
                ['13', '--', '14', '--', '15', '--'],
                ['--', '17', '--', '18', '--', '19'],
                ['21', '--', '22', '--', '23', '--']
            ])

        for square, piece in self._state[Color.BLACK].items():
            x, y = self.squareToPosition[square]
            board[x][y] = 'bb' if piece is Piece.PAWN else 'BB'

        for square, piece in self._state[Color.WHITE].items():
            x, y = self.squareToPosition[square]            
            board[x][y] = 'ww' if piece is Piece.PAWN else 'WW'

        # if self.turn is Color.BLACK:
        #     board = np.flipud(board)
        #     board = np.fliplr(board)

        print(board)
