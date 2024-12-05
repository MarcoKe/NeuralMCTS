This is a research repository for neurally guided Monte Carlo tree search (AlphaGo-esque algorithms) for single-player problems (although the focus is on the job shop problem). 

It allows for the configuration of a wide variety of algorithmic variation in the different MCTS phases and more. For details, please consult my [PhD Thesis](https://publications.rwth-aachen.de/record/990042). Further information can also be found in my [systematic literature review on neural MCTS](https://link.springer.com/article/10.1007/s10489-023-05240-w) as well as [my paper on neural MCTS for the JSP](https://www.scitepress.org/Papers/2024/123117/123117.pdf). 

# Execution 

All code is written to be executed from the main directory. 

To start an experiment, simply execute 
```python main.py exp_name```

where ```exp_name``` is the name of an experiment configuration file in ```data/config/experiments``` without the file ending. 

For instance: ```python main.py exp_001```
