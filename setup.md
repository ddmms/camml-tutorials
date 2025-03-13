# Introduction to Work Environment

## Access the cloud

We will use STFC cloud environment
<https://training.jupyter.stfc.ac.uk>. It runs a custom baked docker
image of Ubuntu Noble Numbat

Use the username given at registration and following instructions to
setup your instance.

### Create account and login

1.  Go to training.jupyter.stfc.ac.uk DO NOT CLICK on sign in!

   ![landing page](figs/01-startup-screen.webp)

2.  Signup: click on **Signup** then use the username given and choose
    password and click **Create User**

   ![landing page](figs/02-signup.webp)

Authorization happens behind the scenes if successful you will see
something like.

![landing page](figs/03-success.webp)

1.  Login with the credentials from above


   ![landing page](figs/04-login.webp)

You shall see something like this, if all ok,

![landing page](figs/05-success.webp)

or instance already started. see below.

### Create instance

In the list you shall see ML Autumn School 2023, select it and click start.

Once you click start will spawn the new VM machine, shall take 2 min or
so but sometimes can be faster or slower, which exists for 24h by
default and has a persistent home directory associated with your user.


   ![landing page](figs/06-instance.webp)

If you click desktop you will get a minimalistic desktop environment

   ![landing page](figs/07-desktop.webp)

you can see a video of [the process](https://youtu.be/0D6qKeVWuYk)

### Stop instance and update

if things go wrong or you need to create an instance with an updated
image you need to follow the following steps.

1.  get the hub settings: File -\> Hub Control Panel

   ![landing page](figs/01-hub.webp)

2.  stop the instance

    stop the instance by clicking on the "Stop My Server" button then
    once stopped you can click Logout.

   ![landing page](figs/02-stop.webp)


3.  logout and create a new instance as above. This will use the latest
    version of the image.

[a video of the process](https://youtu.be/R0rEl9DbG3k)


## Obtain exercises

open a terminal

``` bash
cd
git clone https://gitlab.com/cam-ml/camml.git WORKSHOP
```

a WORKSHOP folder will appear on the left hand side and now you can
navigate inside it and find the relevant notebook of the day.

> checkout WORKSHOP

## Browsers

**Mozilla Firefox** is installed on the machine.

## Compilers

The **GNU** toolchain is used throughout the summer school and are
available at the unix prompt.

-   **gcc**: the C/C++ compiler
-   **gfortran**: the fortran compiler (it assumes f77 and f95 for `*`.f
    and `*`.f90 respecively). Some of the codes may be in fixed format
    requiring the compiler flag -ffixed-form.
-   **python3** is available on the machine, use python3, be aware that
    python will give you python2.

## Plotting Packages

Two graphics packages are available for plotting graphs: **gnuplot** and
**xmgrace**. You can also use matplotlib from python.

## Molecular Graphics Packages

**VMD** is the basic viewer for use in the summer school. **Jmol** is
also available. In order to use Jmol type *jmol.sh* on the command line.
**VESTA** and **Ovito**

## Editors

There are several editors available. You should choose whichever you are
confortable with.

-   **vi** the venerable UNIX screen mode editor.
-   **vim** the improved venerable UNIX screen mode editor.
-   **emacs** probably the commonest full-screen UNIX editor.
-   **gedit** gui editor

## Terminals

When one refers to terminal, console or command line, usually means a
shell window. Gnome Terminal, xterm and uxterm are available, You can
click on the terminal icon to get one in the desktop or in the jupyter
hub.
