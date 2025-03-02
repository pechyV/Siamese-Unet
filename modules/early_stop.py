class EarlyStopping:
    def __init__(self, patience, min_delta=0.001):
        self.patience = patience  # Počet epoch bez zlepšení, po kterých trénování zastavíme
        self.min_delta = min_delta  # Minimální změna v ztrátě, která bude považována za zlepšení
        self.best_loss = float('inf')  # Nejlepší dosažená validační ztráta
        self.counter = 0  
        self.early_stop = False  

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss  # Aktualizace nejlepší ztráty
            self.counter = 0  # Resetování počítadla
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Pokud je počítadlo větší než trpělivost, trénování se zastaví
