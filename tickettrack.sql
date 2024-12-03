-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Dec 03, 2024 at 05:44 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `tickettrack`
--

-- --------------------------------------------------------

--
-- Table structure for table `notifications`
--

CREATE TABLE `notifications` (
  `n_id` int(11) NOT NULL,
  `message` text NOT NULL,
  `date` date NOT NULL DEFAULT current_timestamp(),
  `status` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notifications`
--

INSERT INTO `notifications` (`n_id`, `message`, `date`, `status`) VALUES
(1, 'Person with image uploads\\face_20241201_010714.jpg is not verified.', '2024-12-01', 'sent'),
(2, 'Person with image uploads\\face_20241201_010714.jpg is not verified.', '2024-12-01', 'sent'),
(3, 'Person with image uploads\\face_20241201_010714.jpg is not verified.', '2024-12-01', 'sent'),
(4, 'Person with image uploads\\face_20241201_010714.jpg is not verified.', '2024-12-01', 'sent'),
(5, 'Person with image uploads\\face_20241201_010714.jpg is not verified.', '2024-12-01', 'sent'),
(6, 'Person with image uploads\\face_20241201_010714.jpg is not verified.', '2024-12-01', 'sent'),
(7, 'Person with image uploads\\face_20241201_011842.jpg is not verified.', '2024-12-01', 'sent'),
(8, 'Person with image uploads\\face_20241201_011842.jpg is not verified.', '2024-12-01', 'sent'),
(9, 'Person with image uploads\\face_20241201_011842.jpg is not verified.', '2024-12-01', 'sent'),
(10, 'Person with image uploads\\face_20241203_100322.jpg is not verified.', '2024-12-03', 'sent');

-- --------------------------------------------------------

--
-- Table structure for table `passengers`
--

CREATE TABLE `passengers` (
  `p_id` int(11) NOT NULL,
  `profile` text NOT NULL,
  `source` text NOT NULL,
  `destination` text NOT NULL,
  `fare` bigint(50) NOT NULL,
  `mode` varchar(255) NOT NULL,
  `status` text NOT NULL,
  `pdate` datetime(6) NOT NULL DEFAULT current_timestamp(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `passengers`
--

INSERT INTO `passengers` (`p_id`, `profile`, `source`, `destination`, `fare`, `mode`, `status`, `pdate`) VALUES
(1, 'uploads\\face_20241201_010714.jpg', 'Naganahalli, Bangalore South, Bengaluru Urban, Karnataka, 562130, India', 'Jyothi', 30, 'cash', 'completed', '2024-12-01 01:07:14.572313'),
(2, 'uploads\\face_20241201_010755.jpg', 'Naganahalli, Bangalore South, Bengaluru Urban, Karnataka, 562130, India', 'Jyothi', 30, 'cash', 'completed', '2024-12-01 01:07:55.469522'),
(3, 'uploads\\face_20241201_011014.jpg', 'Naganahalli, Bangalore South, Bengaluru Urban, Karnataka, 562130, India', 'Jyothi', 30, 'cash', 'completed', '2024-12-01 01:10:14.218342'),
(4, 'uploads\\face_20241201_011842.jpg', '', 'Jyothi', 30, 'cash', 'verified', '2024-12-01 01:18:42.705452'),
(5, 'uploads\\face_20241203_100322.jpg', '', 'Jyothi', 30, 'cash', 'completed', '2024-12-03 10:03:22.500915');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `notifications`
--
ALTER TABLE `notifications`
  ADD PRIMARY KEY (`n_id`);

--
-- Indexes for table `passengers`
--
ALTER TABLE `passengers`
  ADD PRIMARY KEY (`p_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `notifications`
--
ALTER TABLE `notifications`
  MODIFY `n_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `passengers`
--
ALTER TABLE `passengers`
  MODIFY `p_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
