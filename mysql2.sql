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

select distinct t1.Num from Logs t1, Logs t2, Logs t3
where t1.id = t2.id - 1 and t2.id = t3.id - 1 and t1.Num = t2.Num and t2.Num = t3.Num


select distinct t123.Num1 as ConsecutiveNums from (
select t12.Num1 as Num1, t12.Num2 as Num2, t3.Num as Num3 from 
(select t2.id as id, t1.Num as Num1, t2.Num as Num2 from Logs t1
join Logs t2 on t1.id = t2.id - 1) t12
join Logs t3 on t12.id = t3.id - 1
) t123
where t123.Num1 = t123.Num2 and t123.Num2 = t123.Num3

/*185 Department top 3 salaries*/
select Department, Employee, Salary from (
select t2.Name as Department, t1.name as Employee, t1.Salary as Salary, dense_rank() over (partition by t1.DepartmentId order by t1.Salary desc) as salary_rank from Employee t1
join Department t2 on t1.DepartmentId = t2.Id
) tmp
where tmp.salary_rank < 4
order by tmp.salary_rank


/*626 exchange seats*/
select  
case
when id % 2 = 1 and id = (select count(*) from seat) then id
when id % 2 = 0 then id - 1
when id % 2 = 1 and id < (select count(*) from seat) then id + 1
end as id, student
from seat
order by id asc 


/*569 Median Employee Salary*/
with c as
(select id,
company,
salary,
row_number() over(partition by company order by salary) rno,
count(*) over(partition by company) cnt
from employee
)
select c.id, c.company, c.salary
from c
where c.rno in (ceil(c.cnt/2), c.cnt/2+1);






