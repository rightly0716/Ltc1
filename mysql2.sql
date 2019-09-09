/* Write your MySQL query statement below */

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

# 180 consecutive numbers
select distinct t1.Num from Logs t1, Logs t2, Logs t3
where t1.id = t2.id - 1 and t2.id = t3.id - 1 and t1.Num = t2.Num and t2.Num = t3.Num


select distinct t123.Num1 as ConsecutiveNums from (
	select t12.Num1 as Num1, t12.Num2 as Num2, t3.Num as Num3 from 
	(select t2.id as id, t1.Num as Num1, t2.Num as Num2 from Logs t1
	join Logs t2 on t1.id = t2.id - 1) t12
	join Logs t3 on t12.id = t3.id - 1
) t123
where t123.Num1 = t123.Num2 and t123.Num2 = t123.Num3


select distinct t123.Num as ConsecutiveNums from (
	select t1.Num from Logs t1
	join Logs t2 on t1.id = t2.id - 1
	join Logs t3 on t1.id = t3.id - 2
	where t1.Num = t2.Num and t2.Num = t3.Num
) t123
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

/*569 Median Employee Salary !!!*/
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


/* 615 Average Salary: Dept vs Company*/
with t1 as (
select left(ts.pay_date) as pay_month, ts.employee_id as employee_id, te.department_id as department_id, ts.amount as amount
from Salary ts
join employee te
on ts.employee_id = te.employee_id
)
with t2 as (
	select pay_month, department_id, avg(amount) as amount_avg
	from t1
	group by department_id
)
select pay_month, department_id, 
case 
	when amount_avg > (select avg(amount) from t1) then 'higher'
	when amount_avg < (select avg(amount) from t1) then 'lower'
	else 'same'
end as comparison
from t2


/* 570 Managers with at least 5 direct reports */
select Name from Employee 
join (select ManagerId
from Employee
group by ManagerId
having count(*) > 5) t1
on Employee.id = t1.ManagerId


/* 579 Cumulative Salary of an employee in past 3 months but the most recent */
select t1.Id, t1.Month, sum(t1.Salary) as Salary from (
select Id, Month, Salary, rank() over (partition by Id order by Month desc) as no_month
from Employee
) t1
where t1.no_month > 1 and t1.no_month < 5

/* 608 Tree nodes */
select t1.id as id, 
if (t1.p_id is Null, 'Root', if(t1.id in (select p_id from tree), 'Inner', 'Leaf')) as Type
from tree t1

/* 610 Triangle Judgement */
select x, y, z, 
if (x + y < z and x + z < y and y + z < x, 'Yes', 'No') as triangle
from triangle


/* 183 Customers who never order */
select Name from Customers
where Id not in (select CustomerId from Orders)

/* Delete duplicate emails */
select Id, Email from (
select Id, Email, rank() over (partition by Email order by Id asc) as no_email
) t1
where t1.no_email = 1

/* Trips and Users */
with table1 as (
select Client_Id, Driver_Id, Status, Request_at, u1.Banned as client_ban, u2.Banned as driver_ban from Trips t1
join Users u1 on t1.Client_Id = u1.Users_Id
join Users u2 on t1.Client_Id = u2.Users_Id
where client_ban = 'No' and driver_ban = 'No' and Request_at between '2013-10-01' and '2013-10-03'
)
select Request_at as Day, round(sum(if(Status = 'complete', 0, 1))/count(*), 2) as 'Cancellation Rate'
from table1
group by Request_at


/* Find median given frequency of numbers??? */
/*构造中间表t，包含列Number, Frequency, AccFreq（累积频率）, SumFreq（频率求和）, AccFreq范围在[SumFreq / 2, SumFreq / 2 + Frequency]的Number均值即为答案。

因为，AccFreq本身介于[SumFreq / 2, SumFreq / 2 + 1]之间
或者上一行的AccFreq <= SumFreq / 2 并且 当前行的AccFreq > SumFreq / 2 + 1
*/
with cumsum_t as (
select n1.id, n1.Number, n1.Frequency, sum(n2.Number) as cumFreq from Number n1
join Number n2 on n1.id >= n2.id
group by n1.id
order by n1.id)


select avg(n.Number) as median
from Number n
where n.Frequency >= abs(
	(select sum(n1.Frequency) from Number n1 where n1.Number >= n.Number) -
	(select sum(n1.Frequency) from Number n1 where n1.Number <= n.Number)))






