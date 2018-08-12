https://sqlzoo.net/

https://livesql.oracle.com/apex/livesql/file/index.html

https://www.w3resource.com/sql-exercises/

https://www.hackerrank.com/domains/sql/select

SELECT game.mdate, 
       game.team1, 
       SUM(CASE WHEN goal.teamid = game.team1
           THEN 1
           ELSE 0
           END) AS score1,
       game.team2,
       SUM(CASE WHEN goal.teamid = game.team2
           THEN 1
           ELSE 0
           END) AS score2
FROM game
JOIN goal
ON (game.id = goal.matchid)
GROUP BY game.id
ORDER BY game.mdate, goal.matchid


SELECT yr,COUNT(title) FROM
  movie JOIN casting ON movie.id=movieid
         JOIN actor   ON actorid=actor.id
where name='John Travolta'
GROUP BY yr
HAVING COUNT(title)=(SELECT MAX(c) FROM
(SELECT yr,COUNT(title) AS c FROM
   movie JOIN casting ON movie.id=movieid
         JOIN actor   ON actorid=actor.id
 where name='John Travolta'
 GROUP BY yr) AS t
)


select yr, num_movies from 
(select movie.yr, count(*) as num_movies
from movie
join (select actor.name, casting.movieid from casting 
join actor on casting.actorid = actor.id 
where actor.name = 'John Travolta') c 
on movie.id = c.movieid
group by yr) tmp
where tmp.num_movies = (
select max(num_movies) from (select movie.yr, count(*) as num_movies
from movie
join (select actor.name, casting.movieid from casting 
join actor on casting.actorid = actor.id 
where actor.name = 'John Travolta') c 
on movie.id = c.movieid
group by yr) tmp
)


select c.name from
(select actor.name, count(*) as num_star from actor
join casting on actor.id = casting.actorid
where ord = 1
group by actor.name) c
order by c.name
where c.num_star > 30


select dept.name, count(teacher.name) from teacher
right join dept
on teacher.dept = dept.id
group by dept.name

SELECT a.company, a.num, stopa.name, stopb.name
FROM route a JOIN route b ON
  (a.company=b.company AND a.num=b.num)
  JOIN stops stopa ON (a.stop=stopa.id)
  JOIN stops stopb ON (b.stop=stopb.id)
WHERE stopa.name='Craiglockhart'

select distinct t1.num, t1.company, t1.endstop, t2.num, t2.company from
(select distinct comba1.num, comba1.id, comba1.company, comba1.name as startstop, combb1.name as endstop
from (select stops.name, stops.id, route.company, route.num 
from stops join route on stops.id = route.stop) comba1
join (select stops.name, route.company, route.num 
from stops 
join route on stops.id = route.stop) combb1
on comba1.num = combb1.num and comba1.company = combb1.company) t1
join (select distinct comba.num, comba.id, comba.company, comba.name as startstop, combb.name as endstop
from (select stops.name, stops.id, route.company, route.num 
from stops join route on stops.id = route.stop) comba
join (select stops.name, route.company, route.num 
from stops 
join route on stops.id = route.stop) combb
on comba.num = combb.num and comba.company = combb.company) t2
on t1.endstop = t2.startstop
where t1.startstop = 'Craiglockhart' and t2.endstop = 'Sighthill'
ORDER BY LENGTH(t1.num), t1.num, t1.endstop, t1.id


###############################################################
#### functions:
###############################################################
1. COALESCE():
Return the first non-null expression in a list:

select teacher.name, coalesce(dept.name, 'None') from teacher
left join dept
on teacher.dept = dept.id

2. CASE:
#CASE allows you to return different values under different #conditions.
#If there no conditions match (and there is not ELSE) then NULL is returned.
#  CASE WHEN condition1 THEN value1 
#       WHEN condition2 THEN value2  
#       ELSE def_value 
#  END 
  
SELECT name, population
      ,CASE WHEN population<1000000 
            THEN 'small'
            WHEN population<10000000 
            THEN 'medium'
            ELSE 'large'
       END  as ee
  FROM bbc
  
  
select t1.lat_n from (
select LAT_N from station
order by lat_n limit 250
) t1
order by t1.lat_n desc limit 1



## WIndow functions
SELECT start_terminal,
       duration_seconds,
       SUM(duration_seconds) OVER
         (PARTITION BY start_terminal) AS start_terminal_total
  FROM tutorial.dc_bikeshare_q1_2012
 WHERE start_time < '2012-01-08'
 
 #Write a query modification of the above example query that shows the duration of each ride as a percentage of the total time accrued by riders from each start_terminal
 
 SELECT start_terminal,
       duration_seconds,
       SUM(duration_seconds) OVER
         (PARTITION BY start_terminal) AS start_terminal_total,
		(100 * duration_seconds/sum(duration_seconds)) over (partition by start_terminal) as pc_start_terminal
  FROM tutorial.dc_bikeshare_q1_2012
 WHERE start_time < '2012-01-08'


Ads
advertiser_id	ad_id	spend	Date	...

Conversions
ad_id	user_id	conversion$	Date	...



select advertiser_id , count(1) as total_spend_per_advertise
from Ads
group by advertiser_id //此处我面试的时候用的windowing 而非groupby，他就问我为什么，我说是在big data习惯，因为spark groupby有时候会比windowing慢
having date_diff(current_timestamp(),date) beween 1 and 30;

select a.count_of_advertisers_who_has_conversions/b.total_count_of_advertisers as percentage_of_advertisers_who_got_conversions
from 
(select count(distinct advertiser_id) as count_of_advertisers_who_has_conversions
from Ads
join conversions
on Ads.id=conversions.id) a, //此处问我逗号是啥意思，我说代表两个table的分隔，这里产生了cartesian join, 但是因为两个表格都只有一个数字，也无所谓，没必要单独生成table再计算。
(select count(distinct advertiser_id) as total_count_of_advertisers
from Ads
)b;


/* 
*/
table：  member_id|company_name|year_start

Q1: count members who ever moved from Microsoft to Google?

SELECT count(distinct t1.member_id)
FROM table t1, table t2 on t1.member_id = t2.member_id
WHERE t1.year_start < t2.year_start
AND t1.company_name = "microsoft" AND t2.company_name ="google"; 


Q2:  count members who directly moved from Microsoft to Google? (Microsoft -- Linkedin -- Google doesn't count)

SQL: 

with CTE as (select *, rank () over (partition by member_id order by year_start) as rank)
. from: 1point3acres.com/bbs 
select COUNT(distinctc1.member_id) 
from CTE c1 join CTE c2
on c1.member_id = c2.member_id
WHERE c1.company_name ='microsoft' AND c2.company_name ='google'
AND c1.rank+1 = c2.rank


/* RUnning sum*/
SELECT
  CustomerID,
  TransactionDate,
  Price,
  SUM(Price) OVER (PARTITION BY CustomerID ORDER BY TransactionDate) AS RunningTotal
FROM
  dbo.Purchases
  
  
  /*
  */
  3. SQL coding

table name: article_views. 1point3acres.com/bbs

date  viewer_id  article_id  author_id    
2017-08-01  123  456 789                 
2017-08-02  432  543 654
2017-08-01  789  456 789
2017-08-03  567  780 432

How many article authors have never viewed their own article? 

How many members viewed more than one article on 2017-08-01? 

select distinct author_id
from article_views
where author_id not in
(
select distinct author_id
from article_views
where author_id = viewer_id 
)

select count(distinct viewer_id)
from article_views
where date = '2017-08-01'
group by viewer_id
having count(distinct article_id) > 1


/* 2. 给一个flight的table，有depature city和 arrival city，求unique的不论顺序的 组合  

比如 depature, arrival
        A             B. 
        B             A
		结果只出现 A B。
*/
select distinct t1.A, t1.B from t t1
left join t t2
on t1.A=t2.B and t1.B=t2.A
where t2.A is Null and t2.B is NULL