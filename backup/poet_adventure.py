from backup.poetic_duels import PoeticDuelSystem
import json
import random

class PoetGame:
    def __init__(self):
        with open("entities/poets/taliesin.json") as f:
            self.player = json.load(f)
    
    def start(self):
        print(f"Welcome, {self.player['name']} the {self.player['culture'].title()} Bard!")
        self.main_menu()

    def load_opponents(self):
        with open("entities/poets/opponents.json") as f:
            return json.load(f)["opponents"]

    def start_duel(self):
        opponent = random.choice(self.load_opponents())
        duel = PoeticDuelSystem(self.player, opponent)
        print(f"\nFacing {opponent['name']} in poetic combat!")
        duel.start_duel()
        if duel.calculate_awen_flow(self.player) > duel.calculate_awen_flow(opponent):
            print("You win! Gained 3 Awen")
            self.player['awen_capacity'] += 3
        else:
            print("You lost... Gained 1 Awen")
            self.player['awen_capacity'] += 1

    def main_menu(self):
        while True:
            print("\nMain Menu:")
            print("1. Start Poetic Duel")
            print("2. View Poet Profile")
            print("3. Exit")
            
            choice = input("Choose an option: ")
            if choice == "1":
                self.start_duel()
            elif choice == "2":
                self.show_profile()
            elif choice == "3":
                break

    def show_profile(self):
        print(f"\n{self.player['name']} ({self.player['culture'].title()})")
        print(f"Awen: {self.player.get('awen_capacity', 10)}")
        print("Abilities: " + ", ".join(self.player.get('special_abilities', [])))

if __name__ == "__main__":
    game = PoetGame()
    game.start()
