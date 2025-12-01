# The funmixer apptainer container

This document contains instructions for building and running the funmixer container in apptainer. 


## First steps: installing apptainer

### Installing in Linux

Almost all modern Linux distribution have [apptainer](https://apptainer.org/) installed by default. You csan see if it is installed on your system by running

```
$ apptainer --help
```

If you don't have it [follow these instructions](https://apptainer.org/docs/admin/main/installation.html). But those can be a bit difficult to understand, so in Ubuntu do this:

```
$ sudo add-apt-repository -y ppa:apptainer/ppa
$ sudo apt update
$ sudo apt install -y apptainer
$ sudo add-apt-repository -y ppa:apptainer/ppa
$ sudo apt update
$ sudo apt install -y apptainer
```


### Installing in Windows (you need Windows subsystem for linux)

To get apptainer in Windows, you need to install the Windows subsystem for Linux. You then choose a linux distribution. If you choose any relatively recent distrubiton (for example Ubuntu 22.04) it will have apptainer installed by default. 

1. Open a powershell window with admin priviledges. If you are on a managed computer you will need to either ask your IT team for administrator privileges or ask them to install `wsl` on your computer. 
2. Run:
```
wsl install
```
3. Go to the Microsoft Store (search in your start menu) and download Ubuntu (we use Ubuntu 22.04)
4. Make sure you enter a password to this system you remember!
5. You can start the Ubuntu operating system from a powershell window with:
  ```
  > wsl -d Ubuntu-22.04
  ```
6. You are now on a Linux operating system inside your Windows computer. It will have apptainer on it. Verfy with
  ```
  $ apptainer --help
  ```

#### wsl2 troubleshooting

1. wsl sometimes does something funny with a file called `/etc/resolv.conf` if you are running Ubuntu as your wsl operating system. If you are not online it will fail to create this file, and if you are online it will write this file in a way that fails to connect to the internet. Both of which cause various fatal error. 
2. To resolve this issue:
  a. First start your Ubuntu system, then write the file:

  ```
  > wsl -d Ubuntu-22.04
  > sudo nano /etc/resolv.conf
  ```
  b. *Note* if you don't use `sudo`  you won't be allowed to write the file (you will get a `permission denied` error. `sudo` gives you administrator rights, you need to remember the password you set up when you installed the Ubuntu operating system on wsl).
  c. Once you run `nano`, your file should say:
  ```
  generateResolvConf = false
  nameserver 8.8.8.8
  ```

### Installing in MacOS

1. The developers of this software don't use MacOS, so none of this is tested. 
2. However, you should be able to install apptainer, so [follow these instructions](https://apptainer.org/docs/admin/main/installation.html#mac). 


## Now, build the container

The apptainer container is built using instructions in the `funmixer_apptainer.def`. It assumes you have this file in the base directory of funmixer after cloning. That is `/path/to/this/repo/funmixer/`
The `.def` file reads the `requirements.yaml` file from a relative path (in this case just `./`) so don't move either the `.def` file or the `requirements.yaml` file. 

In the directory with `funmixer_container.def` run:

```
$ apptainer build funmixer_container.sif funmixer_container.def
```

This will take some time. Even on a fast internet connection it can take 10 minutes. The good news is you only need to do this once. 


## Now run the container

You should now have a file called `funmixer_container.sif`. This is the actual container. It has a functioning Ubuntu operating system and python environemt on it, so it is not small: 250 Mb.

To run the container, use

```
$ apptainer shell funmixer_container.sif
```

The container has its own local filesystem, funmixer is here: `/opt/src/funmixer`. You can go there to run tests. 

### Mapping a drive to the funmixer container

If you have data in your Linux system somewhere, you can bind this data to the apptainer container like this:

```
$ apptainer shell --bind /my/funmixer/data:/funmixer_data funmixer_container.sif
```

The path before the `:` is the path on your computer, and the path after is the path in the container. So when you are in the container you would navigate to `cd /funmixer_data` to see your data. Note that if you are using Ubuntu inside of Windows Linux Subsystem you will need to make sure your Ubuntu system in WSL can see the directory with your data first (that is, you need to make data visible to WSL when you start it, and then to apptainer when you start that).


## History

* Version 1.0 Simon Mudd 27-Nov-2025. First version of the container.
* Version 1.1 Simon Mudd 01-Dec-2025. Changed relative paths, placed the .def file in main repository, and updated instructions.

