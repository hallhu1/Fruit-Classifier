import kagglehub


def classifier():
    print("Hello from the start of classifier!") 
    # Download latest version
    path = kagglehub.dataset_download("moltean/fruits")

    print("Path to dataset files:", path)

if __name__ == "__main__":
    exit(classifier())
