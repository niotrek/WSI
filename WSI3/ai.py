import time

def minimax(position, depth, maximizing_player, node_count=0):
    best_move = None
    node_count = node_count + 1
    if depth == 0 or position.is_game_over():
        return position.evaluate(), best_move, node_count

    if maximizing_player:
        max_eval = float('-inf')
        for move in position.get_possible_moves():
            evaluation, _, node_count = minimax(move, depth-1, False, node_count)
            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move
        return max_eval, best_move, node_count
    
    else:
        min_eval = float('inf')
        for move in position.get_possible_moves():
            evaluation, _, node_count = minimax(move, depth-1, True, node_count)
            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move
        return min_eval, best_move, node_count


def alpha_beta(position, depth, alpha, beta, maximizing_player, depth_meassure, depths, node_count):
    best_move = None
    depth_meassure = depth_meassure + 1
    node_count = node_count + 1
    if depth == 0 or position.is_game_over():
        depths.append(depth_meassure)
        return position.evaluate(), best_move, depths, node_count
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in position.get_possible_moves():
            evaluation, _, _, node_count = alpha_beta(move, depth-1, alpha, beta, False, depth_meassure, depths, node_count)
            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                depths.append(depth_meassure)
                break
        return max_eval, best_move, depths, node_count
    else:
        min_eval = float('inf')
        for move in position.get_possible_moves():
            evaluation, _, _, node_count = alpha_beta(move, depth-1, alpha, beta, True, depth_meassure, depths, node_count)
            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move
            beta = min(beta, evaluation)
            if beta <= alpha:
                depths.append(depth_meassure)
                break
        return min_eval, best_move, depths, node_count

class position:    
    def __init__(self, board=None):
        if board is None:
            self.board = [' ' for _ in range(9)]
            #self.board = ['X', 'X', 'O', 'O', ' ', ' ', ' ', 'O', ' ']
            #self.board = ['X', 'X', 'O', ' ', 'O', ' ', ' ', 'O', ' ']
        else:
            self.board = board
        self.current_player = 'X'

    def is_game_over(self):
        return ' ' not in self.board or self.check_winner('X') or self.check_winner('O')
    
    def evaluate(self):
        if self.check_winner('X'):
            return 1
        elif self.check_winner('O'):
            return -1
        else:
            return 0
        
    def check_winner(self, player):
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] == player:
                return True
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] == player:
                return True
        if self.board[0] == self.board[4] == self.board[8] == player:
            return True
        if self.board[2] == self.board[4] == self.board[6] == player:
            return True
        return False
    
    def get_possible_moves(self):
        moves = []
        for i in range(9):
            if self.board[i] == ' ':
                new_board = self.board.copy()
                new_board[i] = self.current_player
                new_position = position()
                new_position.board = new_board
                new_position.current_player = 'O' if self.current_player == 'X' else 'X'
                moves.append(new_position)
        return moves
    
    def print_board(self):
        print('-------------')
        for i in range(3):
            print('|', self.board[i*3], '|', self.board[i*3 + 1], '|', self.board[i*3 + 2], '|')
            print('-------------')
    
    def play_game(self):
        while True: 
            self.print_board()     
            if self.current_player == 'X':  
                try:
                    pos = int(input('Enter position (1-9): ')) - 1
                    if pos < 0 or pos > 8 or self.board[pos] != ' ':
                        raise ValueError
                except (ValueError, IndexError):
                    print('Invalid input. Try again.')
                    continue
                self.board[pos] = self.current_player
            else:
                print('Computer is playing')
                time.sleep(1)
                best_move = None
                
                #_, best_move , node_count = minimax(self, 9, False)
                depths = []

                _, best_move, depths, node_count = alpha_beta(self, 9, float('-inf'), float('inf'), False, -1, depths, 0)
                print(depths)
                #print(node_count)
                depths.clear()
                self.board = best_move.board

            self.current_player = 'O' if self.current_player == 'X' else 'X'

            if self.check_winner('X'):
                self.print_board()
                print('Player X wins!')
                break
            elif self.check_winner('O'):        
                self.print_board()
                print('Player O wins!')
                break
            elif self.is_game_over():
                self.print_board()
                print('It\'s a tie!')
                break

if __name__ == '__main__':
    pos= position()
    pos.play_game()
