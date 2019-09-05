/* Write your MySQL query statement below
*/
select p.FirstName as FirstName, p.LastName as LastName, a.City as City, a.State as State from Person p
left join Address a 
on a.PersonID=p.PersonID;


/* 2nd largest salary 
# row_number: 1,2,3,4
# rank: 1,2,2,4
# dense_rank: 1,2,2,3 */
select Salary as SecondHighestSalary from (
select Salary, rank() over (order by Salary desc) as 'SalaryRank' from Employee 
) t1
where t1.SalaryRank = 2;

select max(Salary) as SecondHighestSalary from Employee
where Salary < (select max(Salary) from Employee);

/* 177 Nth highest salary */
select Salary as NthHighestSalary from (
select Salary, rank() over (order by Salary desc) as 'SalaryRank' from Employee 
) t1
where t1.SalaryRank = N;

/* 603 Consecutive seat */
select distinct t1.seat_id from cinema t1
join cinema t2
on abs(t1.seat_id - t2.seat_id) = 1
where t1.free1 = 1 and t2.free2 = 1
order by t1.seat_id;

/* 180 Consecutive numbers */
select distinct t1.Num as ConsecutiveNums 
from Logs t1, Logs t2, Logs t3
where t1.Id = t2.Id - 1 and t2.Id = t3.Id - 1 
and t1.Num = t2.Num and t2.Num = t3.Num;








