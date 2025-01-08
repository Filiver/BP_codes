import matplotlib.pyplot as plt

def plt_exploitations(game_result):
    plt.plot(game_result['exploitations'][0], label='Row')
    plt.plot(game_result['exploitations'][1], label='Column')
    plt.grid(True)
    plt.legend()
    plt.show()