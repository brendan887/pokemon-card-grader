class Pokemon:
    def __init__(self, title, pokemon_set, ungraded, grade_nine, grade_ten):
        self.title = title
        self.pokemon_set = pokemon_set
        self.ungraded = ungraded
        self.grade_nine = grade_nine
        self.grade_ten = grade_ten

    def get_title(self):
        return self.title
    def set_title(self, title):
        self.title = title

    def get_pokemon_set(self):
        return self.pokemon_set
    def set_pokemon_set(self, pokemon_set):
        self.pokemon_set = pokemon_set

    def get_ungraded_price(self):
        return self.ungraded
    def set_ungraded_price(self, loosePrice):
        self.ungraded = loosePrice

    def get_grade_nine_price(self):
        return self.grade_nine
    def set_grade_nine_price(self, grade_nine_price):
        self.grade_nine = grade_nine_price

    def get_grade_ten_price(self):
        return self.grade_ten
    def set_grade_ten_price(self, newPrice):
        self.grade_ten = newPrice

    def print_vals(self):
        print ("Title: {}\nPokemon_set: {}\nUngraded: ${}\nGrade_nine: ${}\nGrade_ten: ${}\n\n".format(self.title, self.pokemon_set, self.ungraded, self.grade_nine, self.grade_ten))