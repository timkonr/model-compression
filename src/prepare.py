from conette import CoNeTTEConfig, CoNeTTEModel

def main():
    config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)

    model.save_pretrained("./model/")

if __name__ == "__main__":
    main()
