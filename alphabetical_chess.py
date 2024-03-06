from itertools import batched
from typing import Self

class Pos:
    def __init__(self, rank: int, file: str):
        self.rank = rank
        self.file = file

    def __str__(self):
        return self.file + str(self.rank)
    
    def __repr__(self):
        return str(self)

    def get_row(self):
        return self.rank - 1

    def get_col(self):
        return ["a", "b", "c", "d", "e", "f", "g", "h"].index(self.file)

    @classmethod
    def unsafe_from_coords(cls, row: int, col: int) -> Self | None:
        if 0 <= row <= 7 and 0 <= col <= 7:
            return cls.from_coords(row, col)
        else:
            return None # This null value must be explicitly considered in any method calling this factory
    
    @classmethod
    def from_coords(cls, row: int, col: int) -> Self:
        return cls(row + 1, ["a", "b", "c", "d", "e", "f", "g", "h"][col])
    
    def __eq__(self, other):
        return self.rank == other.rank and self.file == other.file


class Move:
    def __init__(self, board: "Chessboard", origin: Pos, destination: Pos, promotion: str | None = None):
        self.board = board
        self.origin = origin
        self.destination = destination
        self.promotion = promotion

    def __str__(self):
        # TODO check for castling

        move = ""
        if (piece := self.board.board[self.origin.get_row()][self.origin.get_col()]).upper() != "P":
            move += piece.upper()

        # Search for ambiguity
        ambiguities = list(filter(lambda m: m.destination == self.destination and m.origin != self.origin and self.board.board[m.origin.get_row()][m.origin.get_col()] == piece, self.board.allowed_moves))
        if ambiguities:
            same_col = False
            same_row = False
            for imposter in ambiguities:
                if imposter.origin.get_row() == self.origin.get_row():
                    same_row = True
                if imposter.origin.get_col() == self.origin.get_col():
                    same_col = True
            
            if not (not same_row and same_col):
                move += self.origin.file
            if same_col:
                move += str(self.origin.rank)
                
        if self.board.board[self.destination.get_row()][self.destination.get_col()] is not None:
            if piece.upper() == "P":
                move += self.origin.file
            move += "x"

        move += str(self.destination)

        if self.promotion is not None:
            move += "=" + self.promotion

        if (future := self.board.evaluate_move(self)).in_check():
            if future.allowed_moves:
                move += "+"
            else:
                move += "#"

        return move

    def __repr__(self):
        return str(self)


class Chessboard:
    INITIAL_POSITION = [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"]
    ]
    INITIAL_POSITION.reverse()

    def __init__(self, board=INITIAL_POSITION, white_to_move=True, white_castle_queenside=True, white_castle_kingside=True, black_castle_queenside=True, black_castle_kingside=True, previous_positions=[], *, recursion=True):
        self.board = board
        self.white_to_move = white_to_move
        self.white_castle_queenside = white_castle_queenside
        self.white_castle_kingside = white_castle_kingside
        self.black_castle_queenside = black_castle_queenside
        self.black_castle_kingside = black_castle_kingside
        self.previous_positions = previous_positions
        self.recursion = recursion
        self.allowed_moves = self._allowed_moves()

    def evaluate_move(self, move: Move, *, recursion=True) -> "Chessboard":
        new_board = [row.copy() for row in self.board]
        new_board[move.origin.get_row()][move.origin.get_col()] = None
        if move.promotion is None:
            new_board[move.destination.get_row()][move.destination.get_col()] = self.board[move.origin.get_row()][move.origin.get_col()]
        else:
            new_board[move.destination.get_row()][move.destination.get_col()] = move.promotion.upper() if self.white_to_move else move.promotion.lower()

        # TODO deal with castling
        # TODO deal with en passant

        return Chessboard(new_board,
                          not self.white_to_move,
                          False if move.origin.get_row() == 0 and (move.origin.get_col() == 0 or move.origin.get_col() == 4) else self.white_castle_queenside,
                          False if move.origin.get_row() == 0 and (move.origin.get_col() == 7 or move.origin.get_col() == 4) else self.white_castle_kingside,
                          False if move.origin.get_row() == 7 and (move.origin.get_col() == 0 or move.origin.get_col() == 4) else self.black_castle_queenside,
                          False if move.origin.get_row() == 7 and (move.origin.get_col() == 7 or move.origin.get_col() == 4) else self.black_castle_kingside,
                          [] if self.board[move.destination.get_row()][move.destination.get_col()] is not None
                                or self.board[move.origin.get_row()][move.origin.get_col()].upper() == "P"
                                else self.previous_positions + [self],
                          recursion=recursion)

    def check_draw(self) -> bool:
        # Threefold repetition
        if list(map(lambda cb: cb.board, filter(lambda cb: cb.white_to_move == self.white_to_move, self.previous_positions))).count(self.board) == 2:
            return True

        # Fifty-move rule
        if len(self.previous_positions) == 100:
            return True

        # Stalemate
        if not self.in_check() and not self.allowed_moves:
            return True

        # TODO insufficient material

        return False

    def in_check(self) -> bool:
        # If it were the other player's turn, they could capture the king
        king_position = self._find_king()
        for move in Chessboard(self.board, not self.white_to_move, False, False, False, False, [], recursion=False).allowed_moves:
            if move.destination == king_position:
                return True
        return False
        
    def _find_king(self) -> Pos:
        for r, row in enumerate(self.board):
            for c, piece in enumerate(row):
                if piece == ("K" if self.white_to_move else "k"):
                    return Pos.from_coords(r, c)
        assert False

    def in_checkmate(self) -> bool:
        return self.in_check() and not self.allowed_moves

    def _allowed_moves(self) -> list[Move]:
        moves = []
        for r, row in enumerate(self.board):
            for c, piece in enumerate(row):
                if piece is not None and piece.isupper() == self.white_to_move:
                    match piece.upper():
                        case "R" | "Q" | "B":
                            # Rook moves
                            if piece.upper() != "B":
                                for file in range(c - 1, -1, -1):
                                    if not self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(r, file)):
                                        break
                                for file in range(c + 1, 8):
                                    if not self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(r, file)):
                                        break
                                for rank in range(r - 1, -1, -1):
                                    if not self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(rank, c)):
                                        break
                                for rank in range(r + 1, 8):
                                    if not self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(rank, c)):
                                        break
                            if piece.upper() == "R":
                                continue
                            
                            # Bishop moves
                            for file in range(c - 1, -1, -1):
                                if not self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r - (file - c), file)):
                                    break
                            for file in range(c - 1, -1, -1):
                                if not self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r + (file - c), file)):
                                    break
                            for file in range(c + 1, 8):
                                if not self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r - (file - c), file)):
                                    break
                            for file in range(c + 1, 8):
                                if not self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r + (file - c), file)):
                                    break
                        case "N":
                            # Knight moves
                            for i in range(8):
                                self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(
                                    r + (1 if i > 3 else -1) * (2 if i % 2 else 1),
                                    c + (1 if i % 4 > 1 else -1) * (1 if i % 2 else 2)))
                        case "K":
                            # King moves
                            self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r - 1, c - 1))
                            self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r - 1, c))
                            self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r - 1, c + 1))
                            self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r, c + 1))
                            self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r + 1, c + 1))
                            self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r + 1, c))
                            self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r + 1, c - 1))
                            self._add_move(moves, Pos.from_coords(r, c), Pos.unsafe_from_coords(r, c - 1))
                        case "P":
                            # Pawn moves
                            if piece == "P": # White
                                if r == 6:
                                    # Promotion
                                    if c > 0 and (left_capture := self.board[7][c - 1]) is not None and not left_capture.isupper():
                                        for p in ["B", "N", "R", "Q"]:
                                            self._add_move(moves, Pos.from_coords(6, c), Pos.from_coords(7, c - 1), promotion=p)
                                    if c < 7 and (right_capture := self.board[7][c + 1]) is not None and not right_capture.isupper():
                                        for p in ["B", "N", "R", "Q"]:
                                            self._add_move(moves, Pos.from_coords(6, c), Pos.from_coords(7, c + 1), promotion=p)
                                    if self.board[7][c] is None:
                                        for p in ["B", "N", "R", "Q"]:
                                            self._add_move(moves, Pos.from_coords(6, c), Pos.from_coords(7, c), promotion=p)
                                else:
                                    if r == 4:
                                        # TODO Check for en passant
                                        pass
                                    if c > 0 and (left_capture := self.board[r + 1][c - 1]) is not None and not left_capture.isupper():
                                        self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(r + 1, c - 1))
                                    if c < 7 and (right_capture := self.board[r + 1][c + 1]) is not None and not right_capture.isupper():
                                        self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(r + 1, c + 1))
                                    if self.board[r + 1][c] is None and self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(r + 1, c)) and r == 1 and self.board[3][c] is None:
                                        self._add_move(moves, Pos.from_coords(1, c), Pos.from_coords(3, c))
                            else:
                                if r == 1:
                                    # Promotion
                                    if c > 0 and (left_capture := self.board[0][c - 1]) is not None and left_capture.isupper():
                                        for p in ["B", "N", "R", "Q"]:
                                            self._add_move(moves, Pos.from_coords(1, c), Pos.from_coords(0, c - 1), promotion=p)
                                    if c < 7 and (right_capture := self.board[0][c + 1]) is not None and right_capture.isupper():
                                        for p in ["B", "N", "R", "Q"]:
                                            self._add_move(moves, Pos.from_coords(1, c), Pos.from_coords(0, c + 1), promotion=p)
                                    if self.board[0][c] is None:
                                        for p in ["B", "N", "R", "Q"]:
                                            self._add_move(moves, Pos.from_coords(1, c), Pos.from_coords(0, c), promotion=p)
                                else:
                                    if r == 3:
                                        # TODO Check for en passant
                                        pass
                                    if c > 0 and (left_capture := self.board[r - 1][c - 1]) is not None and left_capture.isupper():
                                        self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(r - 1, c - 1))
                                    if c < 7 and (right_capture := self.board[r - 1][c + 1]) is not None and right_capture.isupper():
                                        self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(r - 1, c + 1))
                                    if self.board[r - 1][c] is None and self._add_move(moves, Pos.from_coords(r, c), Pos.from_coords(r - 1, c)) and r == 6 and self.board[4][c] is None:
                                        self._add_move(moves, Pos.from_coords(6, c), Pos.from_coords(4, c))
        return moves

    def _add_move(self, moves: list[Move], origin: Pos | None, destination: Pos | None, promotion: str | None = None) -> bool:
        """Return true if another move can be added in the same direction, false otherwise.

        Assumes that the piece could legally move to the destination square if the board were large enough."""
        assert origin is not None # We know when it's constructed that it's within bounds
        if destination is None:
            return False # Move out of bounds
        move = Move(self, origin, destination, promotion)
        moving_piece = self.board[move.origin.get_row()][move.origin.get_col()]
        dest_piece = self.board[move.destination.get_row()][move.destination.get_col()]
        
        if dest_piece is not None:
            if dest_piece.isupper() != moving_piece.isupper() and self._check_future(move):
                moves.append(move) # Capture
            return False
        if self._check_future(move):
            moves.append(move) # Move to empty space
            return True
        return False
    
    def _check_future(self, move: Move) -> bool:
        if self.recursion:
            future = self.evaluate_move(move, recursion=False)
            future.white_to_move = not future.white_to_move
            if future.in_check():
                return False # Moved into check (like violating a pin)
        return True


def move_series(moves: list[Move], include_newlines=True) -> str:
    series = ""
    for i, M in enumerate(batched(moves, 2)):
        if len(M) == 2:
            series += f"{i + 1}: {M[0]} {M[1]}" + ("\n" if include_newlines else "  ")
        else:
            series += f"{i + 1}: {M[0]}" + ("\n" if include_newlines else "  ") # Should only ever be the last one, but newline for consistency
    return series


def first_alphabetical_checkmate(board, moves):
    if board.in_checkmate():
        print(move_series(moves))
    if not board.check_draw():
        for move in sorted(board.allowed_moves, key=lambda m: str(m).casefold()):
            first_alphabetical_checkmate(board.evaluate_move(move), moves + [move])
    else:
        print(move_series(moves, include_newlines=False))


if __name__ == "__main__":
    first_alphabetical_checkmate(Chessboard(), [])