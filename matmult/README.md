# Matrix Multiplication with Lattigo

## Setup

Whenever you open a new terminal, run:

```bash
export GOROOT=$HOME/local/go124
export PATH=$GOROOT/bin:$PATH
export GOTOOLCHAIN=local
hash -r
```

## Build and Run

To build the project, run:

```bash
go build
```

Then run the executable:

```bash
./matmult-lattigo -v -trials 2
```

## Changing the Function

In `main.go`, you can choose which function to run by changing line 47.

The available functions are listed in the comment near that line.

## Changing the Matrix Dimension

To change the matrix dimension, go to the function you want to run and modify the `size` variable.
