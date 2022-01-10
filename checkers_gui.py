import sys
import time
import threading
import itertools

import pygame
from pygame.locals import *

class CheckersGui:
    FPS = 15

    # Colors
    WHITE = (255, 255, 255)

    LIGHT_GREEN = (235, 235, 208)
    DARK_GREEN = (119, 148, 85)

    WHITE_PIECES = (248, 248, 248)
    BLACK_PIECES = (86, 83, 82)

    SELECT = (186, 202, 43)
    MOVE = (106, 135, 77)

    KING = (178, 34, 34)

    def __init__(self, env, agent1, agent2):
        self.env = env
        self.players = itertools.cycle((agent1, agent2))

        CheckersGui.CELL_SIZE = 80
        CheckersGui.BOARD_SIZE = self.env.width * CheckersGui.CELL_SIZE

        pygame.init()
        pygame.display.set_caption('English draughts')
        
        self.running = True
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((CheckersGui.BOARD_SIZE, CheckersGui.BOARD_SIZE))

        self.source = None
        self.source_pos = None
        self.moves = []
        self.selected = None

        th = threading.Thread(target=self.update)
        th.start()

        self.game_loop()

        th.join()

        pygame.quit()
        # sys.exit()

    def game_loop(self):
        while self.running:
            self.draw_board()

            for event in pygame.event.get():
                if event.type == MOUSEBUTTONUP:
                    self.handle_mouse_click()
                elif event.type == QUIT:
                    self.running = False
            
            pygame.display.update()
            self.clock.tick(CheckersGui.FPS)

    def update(self):
        agent = next(self.players)

        while self.running and not self.env.is_over:
            if agent is None:
                if self.selected is not None:
                    self.env.step(self.selected)
                    # agent = next(self.players)
                    self._reset()
            else:
                time.sleep(1)
                action_index = agent.best_action(self.env.state, self.env.actions())
                self.env.step(self.env.actions()[action_index])
                # agent.act(0, self.env)
                # agent = next(self.players)

    def draw_board(self):
        self.display.fill(CheckersGui.WHITE)

        board = pygame.Surface((CheckersGui.BOARD_SIZE, CheckersGui.BOARD_SIZE))
        board.fill(CheckersGui.WHITE)

        # font = pygame.font.SysFont('arial', 50)
        colors = itertools.cycle((CheckersGui.LIGHT_GREEN, CheckersGui.DARK_GREEN))

        # square = 1

    	# Draw board
        for square_x in range(0, 8):
            for square_y in range(0, 8):
                color = next(colors)
                pos_x, pos_y = self._board_to_screen((square_x, square_y))

                if self.source_pos == (pos_x, pos_y):
                    color = CheckersGui.SELECT

                rect = Rect(pos_y, pos_x, CheckersGui.CELL_SIZE, CheckersGui.CELL_SIZE)
                pygame.draw.rect(board, color, rect)

                # if (x + y) % 2 == 1:
                #     text = font.render(str(square), True, CheckersGui.LIGHT_GREEN)

                #     center_l = (CheckersGui.CELL_SIZE - text.get_rect().width) / 2
                #     center_t = (CheckersGui.CELL_SIZE - text.get_rect().height) / 2

                #     board.blit(text, (pos_y + center_l, pos_x + center_t))

                #     square += 1

            next(colors)

        # Draw pieces
        pawn_radius = 30
        king_radius = 10

        for color, pieces in self.env._state.items():
            color = CheckersGui.WHITE_PIECES if color.value == 1 else CheckersGui.BLACK_PIECES

            for square, piece in pieces.items():
                pos_x, pos_y = self._board_to_screen(square, square=True, center=True)

                pygame.draw.circle(board, color, (pos_y, pos_x), pawn_radius)

                if piece.value == 2:
                    pygame.draw.circle(board, CheckersGui.KING, (pos_y, pos_x), king_radius)

        # Draw moves
        move_radius = 15
        for move in self.moves:
            pos_x, pos_y = self._board_to_screen(move, square=True, center=True)
            pygame.draw.circle(board, CheckersGui.MOVE, (pos_y, pos_x), move_radius)

        self.display.blit(board, board.get_rect())
    
    def handle_mouse_click(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        square = self._square_clicked(mouse_x, mouse_y)

        if square is None:
            self._reset()

        if self.source is None:
            self._reset(square)
        else:
            if square not in self.moves:
                self._reset(square)
            else:
                self.moves = []
                self.selected = (self.source, square, self._get_captures(square))

    def _square_clicked(self, x, y):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        square_x, square_y = int(mouse_x / CheckersGui.CELL_SIZE), int(mouse_y / CheckersGui.CELL_SIZE)

        square = None
        
        try:
            square = self.env.positionToSquare[(square_y, square_x)]
        except KeyError:
            return None
        
        return square
    
    def _reset(self, square=None):
        self.source = square
        self.source_pos = self._board_to_screen(square, square=True) if square is not None else None
        self.moves = self._update_moves() if square is not None else []
        self.selected = None

    def _board_to_screen(self, coord, square=False, center=False):
        square_x, square_y = -1, -1

        if square:
            try:
                square_x, square_y = self.env.squareToPosition[coord]
            except KeyError:
                return square_x, square_y
        else:
            square_x, square_y = coord

        if center:
            return (1 + 2 * square_x) * CheckersGui.CELL_SIZE / 2, (1 + 2 * square_y) * CheckersGui.CELL_SIZE / 2
        else:
            return square_x * CheckersGui.CELL_SIZE, square_y * CheckersGui.CELL_SIZE  

    def _update_moves(self):
        moves = []

        for start, move, jumped in self.env.actions():
            if self.source != start:
                continue

            moves.append(move)

        return moves

    def _get_captures(self, square):
        captures = []

        for start, move, jumped in self.env.actions():
            if self.source != start:
                continue

            if move != square:
                continue

            captures += jumped

        return captures
