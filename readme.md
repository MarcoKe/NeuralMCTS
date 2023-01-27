# How to git

You do not have a personal branch in which you work all the time. 

If you want to implement a specific feature, fix a bug, etc., you create a new branch for this one thing. 
While developing, regularly push to this branch. Once you are done developing and have **tested** your changes,
you create a merge request for your branch. I will review the changes
and integrate them into the main branch. 

This branch is now done. You do not do anything else with it. 
If you start something new, create a new branch. 
The starting point for all new branches is the main branch. 

Try to keep your changes to a local level as much as possible. 
Changes that affect large portions of the code should be discussed first. 

# Testing 

Code should be tested with unit tests. We use the pytest package to do this. 
Simply create a test in an appropriate place in the tests/ directory. 
You can run the tests from the main directory with: ```python -m pytest```