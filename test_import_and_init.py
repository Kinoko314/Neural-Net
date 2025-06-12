import sys
import os

# Add the parent directory to sys.path to allow relative import of scalable_game_ai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from scalable_game_ai.__main__ import Game, Player, Platform, SimpleNN
    print("Successfully imported Game, Player, Platform, SimpleNN from scalable_game_ai.__main__")

    # Try to instantiate the Game object
    # This will trigger weight loading if files exist, or print messages if not/mismatched
    print("Attempting to instantiate Game()...")
    game_instance = Game()
    print("Successfully instantiated Game()")

    # Check number of AI players and networks
    num_ai_players = len(game_instance.ai_players)
    num_ai_nets = len(game_instance.ai_nets)
    print(f"Number of AI players: {num_ai_players}")
    print(f"Number of AI networks: {num_ai_nets}")

    if num_ai_players == 0:
        print("Warning: No AI players were initialized.")
    elif num_ai_nets == 0:
        print("Warning: No AI networks were initialized.")
    elif num_ai_players != num_ai_nets:
        print(f"Warning: Mismatch between AI players ({num_ai_players}) and networks ({num_ai_nets}).")
    else:
        print(f"AI players and networks initialized consistently: {num_ai_players} each.")

    # Check a sample AI player's network input compatibility (W1 shape)
    if num_ai_players > 0:
        sample_player = game_instance.ai_players[0]
        sample_net = game_instance.ai_nets[0]
        expected_input_size = sample_net.W1.shape[0]

        # Construct a dummy input array based on the expected size
        # This simulates the structure of the input array in the game's update loop
        actual_inputs_array = []
        if expected_input_size >= 1: # los_input 1
            actual_inputs_array.append(sample_player.los_input(30, -1, game_instance.platforms))
        if expected_input_size >= 2: # los_input 2
            actual_inputs_array.append(sample_player.los_input(30, 1, game_instance.platforms))
        if expected_input_size >= 3: # ground_gap_input
            actual_inputs_array.append(sample_player.ground_gap_input(game_instance.platforms))
        if expected_input_size >= 4: # vy / 10.0
            actual_inputs_array.append(sample_player.vy / 10.0)
        if expected_input_size >= 5: # get_direction_to_score_line
            actual_inputs_array.append(sample_player.get_direction_to_score_line(game_instance.score_line_x))

        # If more inputs are expected by W1 than provided, this will be an issue.
        # The test here is more about checking the *currently configured* W1 against *what we try to feed it*.
        # The actual number of features in the input array is 5 now.
        num_actual_inputs = 5

        if expected_input_size == num_actual_inputs:
            print(f"NN W1 input dimension ({expected_input_size}) matches number of features being prepared ({num_actual_inputs}).")
        else:
            print(f"ERROR: NN W1 input dimension ({expected_input_size}) MISMATCHES number of features prepared ({num_actual_inputs}).")

    print("Initialization test script completed.")

except ImportError as e:
    print(f"ImportError: {e}")
    print("This might be due to issues with sys.path or the module structure.")
except Exception as e:
    print(f"An error occurred during Game instantiation or testing: {e}")
