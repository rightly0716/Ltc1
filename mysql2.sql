# Write your MySQL query statement below

select p.FirstName as FirstName, p.LastName as LastName, a.City as City, a.State as State from Person p
left join Address a 
on a.PersonID=p.PersonID;


# 2nd largest salary 
# row_number: 1,2,3,4
# rank: 1,2,2,4
# dense_rank: 1,2,2,3
select Salary as SecondHighestSalary from (
select Salary, rank() over (order by Salary desc) as 'SalaryRank' from Employee 
) t1
where t1.SalaryRank = 2;

select max(Salary) as SecondHighestSalary from Employee
where Salary < (select max(Salary) from Employee);


# 603 consevative seat
select t3.seat_id from 
(select t1.seat_id, t1.free as free1, t2.free as free2 from cinema t1
join cinema t2
on t1.seat_id = t2.seat_id - 1) t3
where t3.free1 = 1 and t3.free2 in (1, NULL);







