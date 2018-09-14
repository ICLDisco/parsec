# Visualizing data streamed from the PaRSEC runtime

To visualize performances and, optionally to control the application resources, this is a list of steps to do. Here is the order things have to be started in order to work:

1. Aggregator
2. Demo server
3. Application
4. GUI

##1. Build PaRSEC

Compile PaRSEC with ```PARSEC_PROF_TRACE``` option turned ```ON```, in order to activate the ```PINS``` modules. ```alperf``` module will be listed during configure, and therefore will be compiled.

The fastest way to try is to compile dplasma with the default precisions.

##2. Starting the aggregator process

Set up these environment variables for the PaRSEC runtime to know where to find the aggregator process:

```{Bash}
export AGGREGATOR_HOSTNAME=<hostname of the machine running the aggregator>
export AGGREGATOR_PORT=<the port on which the aggregator is listenning, P1>
```

Call the aggregator process:

```{Bash}
./aggregator.py -N <nb_processes N>
                -M <nb execution units / argo streams M>
                -P <nb of rows in the process grid P>
                -Q <nb of columns in the process grid Q>
                -p <port number to listen for connection, P1>
```

### (Optional) Using a server to send commands

The application needs to know where is this server, so:

```{Bash}
export SERVER_HOSTNAME=<hostname of the machine running the demo server>
export SERVER_PORT=<demo_server is waiting on port P2>
```

To start the demo server:

```{Bash}
$(DPLASMA_BUILD)/demo_server <number of mpi processes N>
                             <port number to listen for connection, P2>
```
To control the demo server, the format of commands is:

```{Bash}
> i c j
```

* **i** is an integer, the rank of the target MPI rank to control, -1 is a broadcast
* **c** is a character for the command:  
  * n: this command doesn't take the core rank, and will return the statuses of all the cores  
  * s: this command will stop the target core on the target process  
  * w: this command will start the target core on the target process
* **j** is an integer, the rank of the target core to control

##3. Starting the application

Start your favorite application running on top of PaRSEC (+argobots).

```{Bash}
[mpirun -np N]
  ./app [application specific parameters]
        -- --mca mca_pins alperf --mca pins_alperf_events task,flops
```

After starting the application, the aggregator should list the available keys {K} to plot.

##4. Visualizing

In order to plot, you need to run:

```{Bash}
./plot_gui.py -N <nb processes N>
              -M <nb computing threads M>
              -P <nb rows in the process grid P>
              -Q <nb columns in the process grid Q>
              -g <aggregator hostname>
              -a <aggregator port P1>
```

The GUI application is connected to the aggregator, set of plotable keys has been shared and acknowledged with the user. You can ask to plot a key:

```{Bash}
> plot <key_name>
```
