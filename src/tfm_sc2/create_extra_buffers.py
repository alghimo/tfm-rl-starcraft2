import argparse


def main(args: argparse.Namespace):
    print(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("buffer", type=str, nargs="+", help="Buffer files to mix. You should provide at least two files")
    parser.add_argument("output_buffer", type=str, help="Output buffer file. It will contain a new experience replay buffer with the memories from the other buffers, pickled into this file.")
    parser.add_argument("mix_memories_method", type=str, default="balanced", choices=["full", "balanced"], help="How to mix the memories: full = take all memories from each buffer, balanced = take the same number of memories from each buffer")
    args = parser.parse_args()

    main(args)
