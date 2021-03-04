import logging
def main():
    for i in range(3):
        logging.basicConfig(filename=f'result_{i}.log', filemode='w', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
        logging.warning(f"Seen examples from TREC50: 0")
        logging.warning(f"Run {i}")

if __name__ == "__main__":
    main()