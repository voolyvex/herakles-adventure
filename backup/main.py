import json
import random
import sys
from time import sleep as s


def run():
    with open("characters.json") as file:
        character_dict = json.load(file)

    hero = "Herakles"

    def death(monster):
        print("You were killed by " + monster + ".")
        new_game = input("Play again? (y/n): ").strip().lower()
        while new_game not in ["y","n"]:
            new_game = input("Play again? (y/n): ").strip().lower()
        if new_game == "n":
            print("\nGame Over\n")
            sys.exit()
        else:
            run()

    def victory(monster):
        defense = character_dict[hero]["Defense"]
        if monster != "Cerberus":
            print("Wow, you defeated " + monster + ".")
            user_prompt = input("Continue? (y/n): ").strip().lower()
            while user_prompt not in ["y","n"]:
                user_prompt = input("Continue? (y/n): ").strip().lower()
            if user_prompt == "n":
                s(1)
                print("\nGame Over")
                sys.exit()
            elif monster == "Nemean Lion":
                equip_prompt = input("Equip the Lion Skin? (y/n): ").strip().lower()
                while equip_prompt not in ["y","n"]:
                    equip_prompt = input("Equip the Lion Skin? (y/n): ").strip().lower()
                s(1)
                if equip_prompt == "n":
                    print("\nYou think, 'Why wear clothing, I'm Herakles?!'")
                    s(1)
                    print("You leave the lion's skin as is and depart.")
                    s(1)
                elif equip_prompt == "y":
                    character_dict[hero]["Defense"] = 2
                    print("Lion Skin equipped\nDefense +2")
                    s(1)
            elif monster == "Lernaean Hydra":
                s(1)
                print("You severed the immortal head.")
                s(1)
                equip_prompt = input("Dip arrows in poison? (y/n): ").strip().lower()
                while equip_prompt not in ["y","n"]:
                    equip_prompt = input("Dip arrows in poison? (y/n): ").strip().lower()
                if equip_prompt == "n":
                    print("\nYou decide that 'hard mode' is more fun and continue with non-poison arrows.")
                elif equip_prompt == "y":
                    print("Poison Arrows equipped\nAttack Power +6")
                    character_dict[hero]["Attack"][0] = ["Poison Arrows", 15]
        else:
            s(1)
            print("Congratulations! You captured Cerberus.")
            s(1)
            print("King Eurystheus is actually terrified of Cerberus,")
            print("so you release him back to Hades.")
            s(1)
            print("You won the game.")
            if defense == 0:
                s(3)
                print("'Skinless' bonus acheived!")
            if character_dict[hero]["Attack"][0][0] != "Poison Arrows":
                s(3)
                print("'Poisonless' bonus acheived!")
            s(1)
            print("Game by voolyvex")
            sys.exit()

    def both_die(monster):
        s(1)
        print("\nWow you defeated " + monster + ",")
        s(1)
        print("but you also died in the process :(")
        s(1)
        new_game = input("Play again? (y/n): ").strip().lower()
        while new_game not in ["y","n","yes","no"]:
            new_game = input("Play again? (y/n): ").strip().lower()
        if new_game == "n" or new_game == "no":
            print("\nGame Over")
            sys.exit()
        else:
            run()

    def attack():
        hero_attack_randomized = random.randint(0,2)
        if hero_attack_randomized == 0:
            print(hero, "attacks with", character_dict[hero]["Attack"][0][0])
            damage = character_dict[hero]["Attack"][0][1]
        elif hero_attack_randomized == 1:
            print(hero, "attacks with", character_dict[hero]["Attack"][1][0])
            damage = character_dict[hero]["Attack"][1][1]
        elif hero_attack_randomized == 2:
            print(hero, "attacks with", character_dict[hero]["Attack"][2][0])
            damage = character_dict[hero]["Attack"][2][1]
        if random.random() > 0.8:
            damage += random.choice([1,2,3,4])
            print("\nCritical hit!\n")
        elif random.random() < 0.05:
            print(hero, "misses.")
            damage = 0
        return(damage)

    def item():
        boar_meat = int(character_dict[hero]["Item"]["Boar Meat"])
        print(f"Herakles has ({boar_meat}) Boar Meat")
        item_prompt = input("Use item: Boar Meat (y/n) ").strip().lower()
        while item_prompt not in ["y","yes","n","no"]:
            item_prompt = input("Use item: Boar Meat (y/n) ").strip().lower()
        if item_prompt == 'y' or item_prompt == 'yes':
            if boar_meat > 0:
                boar_meat -= 1
                character_dict[hero]["Item"]["Boar Meat"] = boar_meat
                return(22, boar_meat, False)
            elif boar_meat <= 0:
                boar_meat = 0
                print("You have no boar meat.\n")
                return(0, boar_meat, True)
        else:
            return(0, boar_meat, True)
        
        
    def random_attack(boss, dead):
        s(1)
        attack_int = random.randint(0,1)
        if dead == False:
            if attack_int == 0:
                print(boss, "attacks with", character_dict[boss]["Attack"][0][0])
                damage = character_dict[boss]["Attack"][0][1]
            else:
                print(boss, "attacks with", character_dict[boss]["Attack"][1][0])
                damage = character_dict[boss]["Attack"][1][1]
            if random.random() < 0.05:
                print(boss, "misses.")
                damage = 0
        else:
            print("With their last dying breath.....")
            s(1)
            if random.random() < 0.5:
                print(boss, "lunges at you, but you dodge the blow easily.")
                damage = 0
            else:
                if attack_int == 0:
                    print(boss, "attacks with", character_dict[boss]["Attack"][0][0])
                    damage = character_dict[boss]["Attack"][0][1]
                else:
                    print(boss, "attacks with", character_dict[boss]["Attack"][1][0])
                    damage = character_dict[boss]["Attack"][1][1]
                if random.random() < 0.05:
                    print(boss, "misses.")
                    damage = 0
        return(damage)

    def boss_fight(boss):
        dead = False
        s(1)
        hero_health = int(character_dict[hero]["Health"])
        boss_health = int(character_dict[boss]["Health"])
        defense = character_dict[hero]["Defense"]
        print("You are in a boss fight vs", boss)
        s(1)
        print()
        print(hero, "HP: ", hero_health)
        print(boss, "HP: ", boss_health)
        s(1) 
        print("\nEnter a command: (A)ttack or (I)tem")
        while (boss_health > 0) and (hero_health > 0):
            action = input("Choose an action: ").strip().lower()
            while action not in ["attack","a","item","i"]:
                action = input("Choose an action: (A)ttack or (I)tem ").strip().lower()
            if action == "attack" or action == "a":
                attack_damage = attack()
                boss_health -= attack_damage
                s(1)
                print(hero, "inflicts", attack_damage, "damage")
                if boss_health <= 0:
                    dead = True
                hero_damaged = random_attack(boss, dead)
                hero_health = (hero_health - hero_damaged) + defense
                s(1)
                print(boss, "inflicts", hero_damaged, "damage\n")
                if hero_health <= 0:
                    print(hero, "HP: ", 0)
                else:
                    print(hero, "HP: ", hero_health)
                if boss_health <= 0:
                    print(boss, "HP: ", 0)
                else:
                    print(boss, "HP: ", boss_health)

            elif action == "item" or action == "i":
                item_effect, boar_meat, ambush = item()
                hero_health += item_effect
                if hero_health >= 50:
                    hero_health = 50
                s(1)
                print(hero, "is healed +", item_effect, "HP")
                s(1)
                print("You have", boar_meat, "Boar Meat(s) left.\n")
                if ambush == True:
                    print("While checking your inventory")
                    s(1)
                    print("you are ambushed.\n")
                hero_damaged = random_attack(boss, dead)
                hero_health = (hero_health - hero_damaged) + defense
                s(1)
                print(boss, "inflicts", hero_damaged, "damage\n")
                s(1)
                if hero_health <= 0:
                    print(hero, "HP: ", 0)
                else:
                    print(hero, "HP: ", hero_health)
                s(1)
                if boss_health <= 0:
                    print(boss, "HP: ", 0)
                else:
                    print(boss, "HP: ", boss_health)
        character_dict[hero]["Health"] = hero_health
        if (boss_health <= 0) and (hero_health <= 0):
            both_die(boss)
        elif (boss_health <= 0):
            victory(boss)
        else:
            death(boss)


    def main():    
        print("\nYou are Herakles of ancient legend.\nYou have been tasked by King Eurystheus to slay the\nvicious Nemean Lion, defeat the nine-headed Lernaean\nHydra, and capture Cerberus--the guard dog of the underworld.")
        s(2)
        user_prompt = input("\nStart Game (y/n): ").strip().lower()
        while user_prompt not in ["y","n"]:
            user_prompt = input("Start Game (y/n): ").strip().lower()
        if user_prompt == "n":
            print("\nGame Over\n")
            sys.exit()
        else:
            s(1)
            print("\nYou make the long & humbling journey to Nemea.\nThe Nemean Lion snarls, 'Dare thee challenge?!'")
            boss_fight("Nemean Lion")
            s(1)
            print("\nYou travel to the outskirts of Lerna and find the\nadorable lair of the Laernean Hydra.")
            print("One of the nine heads hisses, 'Dieeee.'")
            boss_fight("Lernaean Hydra")
            s(1)
            print("\nFinally, the entrance to Hell. Oh wow look it's Cerberus. Nice doggy.")
            print("\tNiiiiice doggy.'\n")
            boss_fight("Cerberus")

    main()

run()