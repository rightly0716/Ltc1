/* -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:20:56 2018

@author: u586086
http://www.raychase.net/2810
https://byrony.github.io/summary-of-sql-questions-on-leetcode.html
"""
*/*/


/* 175 Combine Two Tables
Table: Person
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| PersonId    | int     |
| FirstName   | varchar |
| LastName    | varchar |
+-------------+---------+
PersonId is the primary key column for this table.


Table: Address
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| AddressId   | int     |
| PersonId    | int     |
| City        | varchar |
| State       | varchar |
+-------------+---------+
AddressId is the primary key column for this table.

Write a SQL query for a report that provides the following information for each 
person in the Person table, regardless if there is an address for each of those people:
FirstName, LastName, City, State
*/

select p.FirstName, p.LastName a.City, a.State from Person p
left outer join Address a
on p.PersonId = a.PersonId;



/*  176 Second Highest Salary  
Write a SQL query to get the second highest salary from the Employee table.
+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+

For example, given the above Employee table, the second highest salary is 200. If there is no second highest salary, then the query should return null. 

*/

select max(Salary) as secondHighestSalary
from Employee
where Salary < (select max(Salary) from employee);



/* 177 Nth Highest Salary 
Write a SQL query to get the nth highest salary from the Employee table.
+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+


For example, given the above Employee table, the nth highest salary where n = 2 is 200. If there is no nth highest salary, then the query should return null. 

*/

sorted(Employee['Salary'], reverse = True)[n]

/* 178 Rank Scores 
Write a SQL query to rank scores. If there is a tie between two scores, both should 
have the same ranking. Note that after a tie, the next ranking number should be 
the next consecutive integer value. In other words, there should be no “holes” between ranks.
+----+-------+
| Id | Score |
+----+-------+
| 1  | 3.50  |
| 2  | 3.65  |
| 3  | 4.00  |
| 4  | 3.85  |
| 5  | 4.00  |
| 6  | 3.65  |
+----+-------+

For example, given the above Scores table, your query should generate the following report (order by highest score):
+-------+------+
| Score | Rank |
| 4.00  | 1    |
| 4.00  | 1    |
| 3.85  | 2    |
| 3.65  | 3    |
| 3.65  | 3    |
| 3.50  | 4    |
+-------+------+
*/

import numpy as np
score = [3.5, 3.65, 4, 3.85, 4, 3.65]
ss = sorted(score, reverse = True)
rank = [1]
pre = 0
next = 1
curr_rank = 1
while next < len(ss):
    if ss[pre] > ss[next]:
        curr_rank = curr_rank + 1
        rank.append(curr_rank)
    else:
        rank.append(curr_rank)
    next = next + 1
    pre = pre + 1
    


/*  180 Consecutive Numbers 
Write a SQL query to find all numbers that appear at least three times consecutively.
+----+-----+
| Id | Num |
+----+-----+
| 1  |  1  |
| 2  |  1  |
| 3  |  1  |
| 4  |  2  |
| 5  |  1  |
| 6  |  2  |
| 7  |  2  |
+----+-----+


For example, given the above Logs table, 1 is the only number that appears consecutively for at least three times. 

*/

nums = [1,1,1,2,3,3,3,2,1,2,2, 1, 1, 1]
pre = 0
next = 1
counts = 2 if nums[pre] == nums[next] else 1
hs = set()
while next < len(nums):
    if nums[pre] == nums[next]:
        counts = counts + 1
    else:
        counts = 1
    if counts == 3:
        hs.add(nums[next])
    next = next + 1
    pre = pre + 1
    

/* 181 Employees Earning More Than Their Managers 
The Employee table holds all employees including their managers. Every employee
 has an Id, and there is also a column for the manager Id.
+----+-------+--------+-----------+
| Id | Name  | Salary | ManagerId |
+----+-------+--------+-----------+
| 1  | Joe   | 70000  | 3         |
| 2  | Henry | 80000  | 4         |
| 3  | Sam   | 60000  | NULL      |
| 4  | Max   | 90000  | NULL      |
+----+-------+--------+-----------+


Given the Employee table, write a SQL query that finds out employees who earn 
more than their managers. For the above table, Joe is the only employee who earns more than his manager.
+----------+
| Employee |
+----------+
| Joe      |
+----------+

*/
select e.Name
from employee e join employee m
on e.ManagerId = m.Id
where e.Salary > m.Salary


/*  182 Duplicate Emails 
Write a SQL query to find all duplicate emails in a table named Person.
+----+---------+
| Id | Email   |
+----+---------+
| 1  | a@b.com |
| 2  | c@d.com |
| 3  | a@b.com |
+----+---------+

For example, your query should return the following for the above table:
+---------+
| Email   |
+---------+
| a@b.com |
+---------+

Note: All emails are in lowercase. 

*/

select Email
from Person
group by Email
Having count(*) > 1;

select Email from (
    select Email, count(*) as ecount 
    from Person
    group by Email
        ) as temp
where temp.ecount > 1;

/* 183 Customers Who Never Order 
Suppose that a website contains two tables, the Customers table and the Orders 
table. Write a SQL query to find all customers who never order anything. 

Table: Customers.
+----+-------+
| Id | Name  |
+----+-------+
| 1  | Joe   |
| 2  | Henry |
| 3  | Sam   |
| 4  | Max   |
+----+-------+

Table: Orders.
+----+------------+
| Id | CustomerId |
+----+------------+
| 1  | 3          |
| 2  | 1          |
+----+------------+

Using the above tables as example, return the following:
+-----------+
| Customers |
+-----------+
| Henry     |
| Max       |
+-----------+

*/
select c.Name
from Customers c
left join Orders o
on c.Id = o.CustomerId
where o.id is NULL;


/* 184 Department Highest Salary 
The Employee table holds all employees. Every employee has an Id, a salary, and there is also a column for the department Id.
+----+-------+--------+--------------+
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
+----+-------+--------+--------------+

The Department table holds all departments of the company.
+----+----------+
| Id | Name     |
+----+----------+
| 1  | IT       |
| 2  | Sales    |
+----+----------+

Write a SQL query to find employees who have the highest salary in each of the departments. For the above tables, Max has the highest salary in the IT department and Henry has the highest salary in the Sales department.
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| Sales      | Henry    | 80000  |
+------------+----------+--------+
*/
select d.Name as Department, e.Name as Employee, e.Salary from Employee e
left join Department d
on e.DepartmentId = d.Id
group by e.DepartmentId
order by e.Salary desc Limit 1;

select d.Name as Department, h.Name as Employee, h.Salary 
from (select * , row_number() over (partition by DepartmentId order by Salary desc from Employee) as seqnum ) h
join Department d
on h.DepartmentId = d.Id
where seqnum < 2;

select d.Name as Department, t.Name as Employee, t.Salary from Department d
inner join (select Id, Name, DepartmentId, max(Salary) as Salary from Employee group by DepartmentId) t
on d.Id = t.DepartmentId;
            

/* 185 Department Top Three Salaries 
The Employee table holds all employees. Every employee has an Id, and there is also a column for the department Id.
+----+-------+--------+--------------+
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
| 5  | Janet | 69000  | 1            |
| 6  | Randy | 85000  | 1            |
+----+-------+--------+--------------+

The Department table holds all departments of the company.
+----+----------+
| Id | Name     |
+----+----------+
| 1  | IT       |
| 2  | Sales    |
+----+----------+

Write a SQL query to find employees who earn the top three salaries in each of the department. For the above tables, your SQL query should return the following rows.
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| IT         | Randy    | 85000  |
| IT         | Joe      | 70000  |
| Sales      | Henry    | 80000  |
| Sales      | Sam      | 60000  |
+------------+----------+--------+
*/
select d.Department, h.Employee, h.Salary from 
(select *, row_number() over (partition by DepartmentId order by Salary desc from Employee) as seqnum) h
join Department d
on h.DepartmentId = d.Id
where h.seqnum < 4;


/* 196. Delete Duplicate Emails
Write a SQL query to delete all duplicate email entries in a table named Person, keeping only unique emails based on its smallest Id.
+----+------------------+
| Id | Email            |
+----+------------------+
| 1  | john@example.com |
| 2  | bob@example.com  |
| 3  | john@example.com |
+----+------------------+

Id is the primary key column for this table.
 For example, after running your query, the above Person table should have the following rows:
+----+------------------+
| Id | Email            |
+----+------------------+
| 1  | john@example.com |
| 2  | bob@example.com  |
+----+------------------+
*/

http://www.cnblogs.com/grandyang/p/5371227.html

/* Why ????? */
DELETE p2 FROM Person p1 JOIN Person p2 
ON p2.Email = p1.Email WHERE p2.Id > p1.Id;

select distinct(id_min) as Id, Email 
from (select min(id) as id_min, Email from Email group by Email) min_Email;

select e.Id, e.Email 
from (select *, row_number() over (partition by Email order by Id) as seqnum from Email) e
where e.seqnum < 2;


/* 197. Rising Temperature
Given a Weather table, write a SQL query to find all dates' Ids with higher 
temperature compared to its previous (yesterday's) dates.
+---------+------------+------------------+
| Id(INT) | Date(DATE) | Temperature(INT) |
+---------+------------+------------------+
|       1 | 2015-01-01 |               10 |
|       2 | 2015-01-02 |               25 |
|       3 | 2015-01-03 |               20 |
|       4 | 2015-01-04 |               30 |
+---------+------------+------------------+

For example, return the following Ids for the above Weather table: +----+
| Id |
+----+
|  2 |
|  4 |
+----+
*/

select t1.Id from Temperature t1
join Temperature t2
on Datediff(t1.Date, t2.Date) = 1
where t1.Temperature > t2.Temperature;


/*  262. Trips and Users
The Trips table holds all taxi trips. Each trip has a unique Id, while Client_Id 
and Driver_Id are both foreign keys to the Users_Id at the Users table. Status 
is an ENUM type of (‘completed’, ‘cancelled_by_driver’, ‘cancelled_by_client’).
+----+-----------+-----------+---------+--------------------+----------+
| Id | Client_Id | Driver_Id | City_Id |        Status      |Request_at|
+----+-----------+-----------+---------+--------------------+----------+
| 1  |     1     |    10     |    1    |     completed      |2013-10-01|
| 2  |     2     |    11     |    1    | cancelled_by_driver|2013-10-01|
| 3  |     3     |    12     |    6    |     completed      |2013-10-01|
| 4  |     4     |    13     |    6    | cancelled_by_client|2013-10-01|
| 5  |     1     |    10     |    1    |     completed      |2013-10-02|
| 6  |     2     |    11     |    6    |     completed      |2013-10-02|
| 7  |     3     |    12     |    6    |     completed      |2013-10-02|
| 8  |     2     |    12     |    12   |     completed      |2013-10-03|
| 9  |     3     |    10     |    12   |     completed      |2013-10-03| 
| 10 |     4     |    13     |    12   | cancelled_by_driver|2013-10-03|
+----+-----------+-----------+---------+--------------------+----------+

The Users table holds all users. Each user has an unique Users_Id, and Role is an ENUM type of (‘client’, ‘driver’, ‘partner’).
+----------+--------+--------+
| Users_Id | Banned |  Role  |
+----------+--------+--------+
|    1     |   No   | client |
|    2     |   Yes  | client |
|    3     |   No   | client |
|    4     |   No   | client |
|    10    |   No   | driver |
|    11    |   No   | driver |
|    12    |   No   | driver |
|    13    |   No   | driver |
+----------+--------+--------+

Write a SQL query to find the cancellation rate of requests made by unbanned clients 
between Oct 1, 2013 and Oct 3, 2013. For the above tables, your SQL query should 
return the following rows with the cancellation rate being rounded to two decimal places.
+------------+-------------------+
|     Day    | Cancellation Rate |
+------------+-------------------+
| 2013-10-01 |       0.33        |
| 2013-10-02 |       0.00        |
| 2013-10-03 |       0.50        |
+------------+-------------------+
*/
select nb.Request_at as Day, round(sum(if(nb.status = 'cancelled_by_driver', 1, 0))/count(*), 2) as 'Cancellation Rate' from 
(select * from Trips t 
 left join Users u 
 on t.Client_Id = u.UsersId where u.Banned = 'No') nb
where nb.Request_at between '2013-10-01' and '2013-10-03'
group by nb.Request_at;

