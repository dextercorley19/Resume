/*
 Dexter corley final exam cpsc 308

 Business problem:
A hotel chain wants to optimize its room reservation system to maximize
 occupancy and revenue. The chain wants to track the availability and
 utilization of its rooms to ensure efficient allocation and pricing.
 By analyzing reservation patterns and occupancy rates, the hotel chain
 can make data-driven decisions to improve room management,
 optimize pricing strategies, and enhance guest satisfaction.


 PART 1: Relational DB Requirements
The relational database schema for a hotel chain must include the following information:
Room information: room number, room type, capacity, price per night, etc.
Guest information: guest ID, name, contact details, etc.
Reservation information: reservation ID, guest ID, room number, check-in date, check-out date, etc.
Payment information: reservation ID, payment amount, payment date, payment method, etc.

Room_Info: room_num, room_type, capacity, price
Guest_Info: guestID, first_name, last_name, email
Reservation_Info: reservationID, guestID, room_num, check-in_date, check-out_date,
Payment_Info: reservationID, payment_amt, payment_date, payment_method


 */

 pragma foreign_keys = on; -- turns on referential integrity

CREATE TABLE room_info(
    roomID INTEGER PRIMARY KEY,
    room_type text,
    capacity integer,
    price integer
);

CREATE TABLE guest_info(
    guestID INTEGER PRIMARY KEY,
    first_name text,
    last_name text,
    email text
);

CREATE TABLE reservation_info(
    reservationID integer primary key,
    check_in_date text,
    check_out_date text,
    guestID integer,
    roomID integer,
    foreign key(guestID) REFERENCES guest_info(guestID) ON delete set null,
    foreign key(roomID) REFERENCES room_info(roomID) ON delete set null
);

CREATE TABLE payment_info(
    payment_ID integer primary key,
    reservation_ID integer,
    payment_amt integer,
    payment_date text,
    payment_method text,
    foreign key(reservation_ID) REFERENCES reservation_info(reservationID) ON delete set null

);


/*
PART 2 Inserting Data


 */


INSERT INTO room_info values(101, "Standard", 2, 100),
                            (201, "Deluxe", 4, 150),
                            (301, "Suite", 2, 200);

INSERT INTO guest_info values(1, "John", "Doe", "john.doe@example.com"),
                             (2, "Jane", "Smith", "jane.smith@example.com"),
                             (3, "Mike", "Johnson", "mike.johnson@example.com"),
                             (4, "Emily", "Davis", "emily.davis@example.com"),
                             (5, "David", "Wilson", "david.wilson@example.com");

INSERT INTO reservation_info values(1, "2023-05-01", "2023-05-05", 1, 101),
                                   (2, "2023-05-02", "2023-05-08", 2, 201),
                                   (3, "2023-05-03", "2023-05-07", 3, 301),
                                   (4, "2023-05-05", "2023-05-10", 4, 101),
                                   (5, "2023-05-08", "2023-05-10", 1, 201);

insert into payment_info values(49, 1, 400, "2023-05-05", "credit card"),
                               (50, 2, 1050, "2023-05-08", "cash"),
                               (51, 3, 900, "2023-05-07", "credit card");

-- PART 3: Solving our Business Problem

-- 1
SELECT RI.roomID, RI.room_type
from reservation_info as R
left join room_info as RI using(roomID)
group by check_in_date AND check_out_date between "2023-05-02" AND "2023-05-05"
having reservationID is null;

-- 2

SELECT guest_info.first_name, SUM(payment_info.payment_amt) AS total_amount
FROM guest_info
JOIN reservation_info ON guest_info.guestID = reservation_info.guestID
JOIN payment_info ON reservation_info.reservationID = payment_info.reservation_ID
GROUP BY guest_info.guestID, guest_info.first_name
HAVING total_amount > 1000
ORDER BY total_amount DESC;

-- 3


SELECT avg(julianday(RI.check_out_date) - julianday(RI.check_in_date)) * R.price as average_revenue, R.room_type
from reservation_info as RI
left join room_info as R on RI.roomID = R.roomID
group by room_type
order by average_revenue;

-- 4
SELECT G.first_name
from guest_info as G
left join reservation_info as RI on G.guestID = RI.guestID
where RI.roomID IN (101, 201)
group by G.guestID, G.first_name
HAVING COUNT(DISTINCT RI.roomID) = 2;

-- 5

SELECT (julianday(RI.check_out_date) - julianday(RI.check_in_date)) * R.price as amt_owed, RI.reservationID
from reservation_info as RI
left join room_info as R on RI.roomID = R.roomID
left join payment_info as PI on RI.reservationID = PI.reservation_ID
where amt_owed > PI.payment_amt OR Pi.payment_amt IS NULL;











