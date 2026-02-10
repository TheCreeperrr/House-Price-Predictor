from sklearn.metrics import r2_score
import train_funcs as tf
import config as cf
import time


def train_and_display(model_name, train_function):
    print(f"{cf.YELLOW}[TRAINING]{cf.RESET} {model_name}...", end="", flush=True)
    time_start = time.time()
    score = r2_score(tf.Y_test, train_function())
    total_time = time.time() - time_start
    print(f"\r{cf.GREEN}[COMPLETE][{total_time:.3f}s]{cf.RESET} {model_name}: {score:.2f}    ")
    return score


def main():
    print(f"{cf.MAGENTA}===== MODEL TESTING: R2 SCORE COMPARISON =====")

    print(f"\nTEST 1: NO OPTIMIZATION")

    curr_name, absolute_max = max(
        ((name, train_and_display(name, func)) for name, func in cf.models.items()), key=lambda x: x[1]
    )

    print(f"{cf.MAGENTA}{curr_name.upper()} WAS THE MOST ACCURATE PERFORMING MODEL WITH A SCORE OF {absolute_max:.2f}!!!{cf.RESET}")

    tf.optimize = True

    #Note: My laptop is not a supercomputer so I had to limit the optimization.
    #You can change the variables in train_funcs.py if you got better specs than me
    print(f"\n{cf.MAGENTA}TEST 2: OPTIMIZED MODELS (NOT ALL ARE OPTIMIZED YET. IN PROGRESS)")
    
    curr_name, absolute_max = max(
        ((name, train_and_display(name, func)) for name, func in cf.models.items()), key=lambda x: x[1]
    )

    print(f"{cf.MAGENTA}{curr_name.upper()} WAS THE MOST ACCURATE PERFORMING MODEL WITH A SCORE OF {absolute_max:.2f}!!!{cf.RESET}")

if __name__=='__main__':
    main()