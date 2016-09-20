

# Python轻量级ORM - peewee

python peewee orm

2014-06-17 11:42 2162人阅读


版权声明：本文为博主原创文章，未经博主允许不得转载。

peewee是一个轻量级的ORM。用的是大名鼎鼎的sqlalchemy内核，采用纯Python编写，显得十分轻便。为了后续方便查看，在这里简单记录下~~

peewee不仅轻量级，还提供了多种数据库的访问，如SqliteDatabase（file or memory）、MYSQLDatabase、PostgresqlDatabase；

接下来就从API上路吧~~~



## 1. fn 函数

For example:

| Peewee expression                        | Equivalent SQL                     |
| ---------------------------------------- | ---------------------------------- |
| fn.Count(Tweet.id).alias('count')        | Count(t1."id") AS count            |
| fn.Lower(fn.Substr(User.username, 1, 1)) | Lower(Substr(t1."username", 1, 1)) |
| fn.Rand().alias('random')                | Rand() AS random                   |
| fn.Stddev(Employee.salary).alias('sdv')  | Stddev(t1."salary") AS sdv         |

Functions can be used as any part of a query:

- select
- where
- group_by
- order_by
- having
- update query
- insert query

```python
# user's username starts with a 'g' or a 'G':
fn.Lower(fn.Substr(User.username, 1, 1)) == 'g'
```



## 2. 表达式支持的操作符

| Comparison | Meaning                                 |
| ---------- | --------------------------------------- |
| ==         | x equals y                              |
| <          | x is less than y                        |
| <=         | x is less than or equal to y            |
| >          | x is greater than y                     |
| >=         | x is greater than or equal to y         |
| !=         | x is not equal to y                     |
| <<         | x IN y, where y is a list or query      |
| >>         | x IS y, where y is None/NULL            |
| %          | x LIKE y where y may contain wildcards  |
| **         | x ILIKE y where y may contain wildcards |

```python
Employee.select().where(Employee.salary.between(50000, 60000))
```

note:

由于sqlite的like函数在默认下是大小写不敏感的，如果想实现大小写搜索，需要用'*'做通配符。



## 3. 实现用户自定义 OPERATOR

Here is how you might add support for modulo and regexp in SQLite:
```python
from peewee import *
from peewee import Expression # the building block for expressions

OP_MOD = 'mod'
OP_REGEXP = 'regexp'

def mod(lhs, rhs):
    return Expression(lhs, OP_MOD, rhs)

def regexp(lhs, rhs):
    return Expression(lhs, OP_REGEXP, rhs)

SqliteDatabase.register_ops({OP_MOD: '%', OP_REGEXP: 'REGEXP'}) # 添加 %、regexp操作

# Now you can use these custom operators to build richer queries:

# users with even ids
User.select().where(mod(User.id, 2) == 0)

# users whose username starts with a number
User.select().where(regexp(User.username, '[0-9].*'))
```



# 4. Joining tables

There are three types of joins by default:

- JOIN_INNER (default)
- JOIN_LEFT_OUTER
- JOIN_FULL

Here are some examples:

```python
User.select().join(Blog).where(
    (User.is_staff == True) & (Blog.status == LIVE))

Blog.select().join(User).where(
    (User.is_staff == True) & (Blog.status == LIVE))
```

subquery:

```python
staff = User.select().where(User.is_staff == True)
Blog.select().where(
    (Blog.status == LIVE) & (Blog.user << staff))
```

补充：在没有通过ForeignKeyField产生外键的多个models中，也可以做join操作，如：

```python
# No explicit foreign key between these models.
OutboundShipment.select().join(InboundShipment, on=(
    OutboundShipment.barcode == InboundShipment.barcode))
```



## 5. 高级查询

To create arbitrarily complex queries, simply use python’s bitwise “and” and “or” operators:

```python
sq = User.select().where(
    (User.is_staff == True) |
    (User.is_superuser == True))
```

The WHERE clause will look something like:

```sql
WHERE (is_staff = ? OR is_superuser = ?)
```

In order to negate an expression, use the bitwise “invert” operator:

```python
staff_users = User.select().where(User.is_staff == True)
Tweet.select().where(~(Tweet.user << staff_users))
```

This query generates roughly the following SQL:

```sql
SELECT t1.* FROM blog AS t1
WHERE
    NOT t1.user_id IN (
        SELECT t2.id FROM user AS t2 WHERE t2.is_staff = ?)
```

Rather complex lookups are possible:

```python
sq = User.select().where(
    ((User.is_staff == True) | (User.is_superuser == True)) &
    (User.join_date >= datetime(2009, 1, 1)))
```

This generates roughly the following SQL:

```sql
WHERE (
    (is_staff = ? OR is_superuser = ?) AND
    (join_date >= ?))
SELECT * FROM user
```



## 6. Aggregating records

```python
# Suppose you have some users and want to get a list of them along with the count of tweets each has made. First I will show you the shortcut:

query = User.select().annotate(Tweet)

# This is equivalent to the following:

query = User.select(
    User, fn.Count(Tweet.id).alias('count')
).join(Tweet).group_by(User)

# You can also specify a custom aggregator. In the following query we will annotate the users with the date of their most rece#nt tweet:

query = User.select().annotate(
    Tweet, fn.Max(Tweet.created_date).alias('latest'))

# Conversely, sometimes you want to perform an aggregate query that returns a scalar value, like the “max id”. Queries like this can be executed by using the aggregate() method:

most_recent_tweet = Tweet.select().aggregate(fn.Max(Tweet.created_date))
```



## 7. Window functions

```python
# peewee comes with basic support for SQL window functions, which can be created by calling fn.over() and passing in your partitioning or ordering parameters.

# Get the list of employees and the average salary for their dept.
query = (Employee
         .select(
             Employee.name,
             Employee.department,
             Employee.salary,
             fn.Avg(Employee.salary).over(
                 partition_by=[Employee.department]))
         .order_by(Employee.name))

# Rank employees by salary.
query = (Employee
         .select(
             Employee.name,
             Employee.salary,
             fn.rank().over(
                 order_by=[Employee.salary])))
```



## 8. 优化 query 语句, 避免 N+1

```python
# We can do this pretty easily:

for tweet in Tweet.select().order_by(Tweet.created_date.desc()).limit(10):
    print '%s, posted on %s' % (tweet.message, tweet.user.username)

# Looking at the query log, though, this will cause 11 queries:
# 1 query for the tweets
# 1 query for every related user (10 total)
# This can be optimized into one query very easily, though:

tweets = Tweet.select(Tweet, User).join(User)
for tweet in tweets.order_by(Tweet.created_date.desc()).limit(10):
    print '%s, posted on %s' % (tweet.message, tweet.user.username)

# Will cause only one query that looks something like this:
```

```sql
SELECT t1.id, t1.message, t1.user_id, t1.created_date, t2.id, t2.username
FROM tweet AS t1
INNER JOIN user AS t2
    ON t1.user_id = t2.id
ORDER BY t1.created_date desc
LIMIT 10
```

