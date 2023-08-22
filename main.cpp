#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

using namespace std;
using namespace cv;

#define COL 4272
#define ROW 2848
#define PI 3.1415926
#define iter 49
#define pixel 0.00519663

class PointPho {
public:
	double x, y;
};
class PointGCP {
public:
	double X, Y, Z;
};
class IOP {
public:
	double x0, y0, fx, fy, dbeta, ds;
};
class EOP {
public:
	double Xs, Ys, Zs, omega, phi, kappa;
};

//Resection
class CResection {
public:
	CResection(IOP iops, EOP eops);
	~CResection();
	int num;//number of control points
	vector<int> id_photos, id_gcps, id_unknown;
	vector<PointPho> photos, left_unknown, right_unknown;
	vector<PointGCP> gcps, gcps_unknown;
	double x0, y0, f;//interior orientation elements
	double Xs, Ys, Zs, omega, phi, kappa;//exterior orientation elements
	double k1, k2, p1, p2;//distortion coefficient

	void readfile_gcps(string filename);
	void readfile_photos(string filename);
	void readfile_unknown(string filename);
	Mat Rotation(double omega, double phi, double kappa);
	//resection
	void cal_resection(string filename);

	//void outer_precision_check();
};

//DLT
class CDLT {
public:
	CDLT(string path_left, string path_right, string path_gcps, string path_unknown);
	~CDLT();
	int num;
	vector<double>m;//mean error for L1~L11 and distortion coefficients
	//vector<double>m_i,m_e;//mean error for interior & exterior orientation parameters
	vector<int> id_photos_left, id_photos_right, id_gcps, id_ctrl, id_unknown;
	vector<PointPho> photos_left, photos_right, pho_ctrl, left_unknown, right_unknown;
	vector<PointGCP> gcps, gcps_ctrl, gcps_unknown;
	IOP iops, iops2;
	EOP eops, eops2;
	Mat L, L2;   //L-left, L2-right, L12~L15(k1 k2 p1 p2)

	void readfile_gcps(string filename);//get gcps
	void readfile_photos(string filename1, string filename2);//get measurement results of image points
	void readfile_unknown(string filename);//get image coordinates of pairs
	Mat Rotation(double omega, double phi, double kappa);//calculate rotation matrix
	void init_ctrl(vector<int> id, vector<PointPho> photos);//initialize control points
	Mat init_L();//initialize L1~L11 & k1 k2 p1 p2
	Mat cal_L(vector<int> id, vector<PointPho> photos, string filename);//iterative solution for L and distortion coefficients
	double Getfx(Mat LL);//calculate fx
	double Getfy(Mat LL);//calculate fy
	IOP GetIOP(Mat LL);//interior orientation elements
	EOP GetEOP(Mat LL);//exterior orientation elements
	void GetElement(vector<int> id, vector<PointPho> photos, string tag);//calculate and output interior & exterior orientation elements & distortion coefficient
	//void outer_precision_check(vector<int> id_photos, vector<PointPho> photos);
	void init_XYZ();//initialize unknown points coordinates
	void cal_XYZ();//calculate accurate coordinates of unknown points
	void XYZ_precision_check();//accuracy assessment of unknown point coordinates
};

//resection
CResection::CResection(IOP iops, EOP eops) {
	x0 = iops.x0;
	y0 = iops.y0;
	f = 25;//mm

	Xs = eops.Xs;
	Ys = eops.Ys;
	Zs = eops.Zs;
	omega = eops.omega;
	phi = eops.phi;
	kappa = eops.kappa;

	k1 = 0;
	k2 = 0;
	p1 = 0;
	p2 = 0;

	num = 50;
};
CResection::~CResection() {
	id_photos.clear();
	id_gcps.clear();
	photos.clear();
	gcps.clear();
};

//read ground coordinates of GCP (transform)
void CResection::readfile_gcps(string filename) {
	ifstream fin(filename);
	int num_gcps, id, flag;
	PointGCP tmp;

	fin >> num_gcps;
	for (int i = 0; i < num_gcps; i++) {
		fin >> id >> tmp.Z >> tmp.X >> tmp.Y >> flag;//transform to photogrammetric coordinate system
		tmp.Z = -tmp.Z;

		gcps.push_back(tmp);
		id_gcps.push_back(id);
	}
	fin.close();
	cout << "GCPs Num:" << num_gcps << endl;
};

//read result of pixel coordinate measurement of gcp
void CResection::readfile_photos(string filename) {
	int id, num_points;
	PointPho tmp;

	ifstream fin(filename);
	fin >> num_points;

	for (int i = 0; i < num_points; i++) {
		fin >> id >> tmp.x >> tmp.y;

		tmp.x -= COL / 2;//pixel
		tmp.y = ROW / 2 - tmp.y;//pixel

		tmp.x = tmp.x * pixel;//mm
		tmp.y = tmp.y * pixel;//mm

		photos.push_back(tmp);
		id_photos.push_back(id);
	}
	fin.close();

	cout << "Read points num:" << num_points << endl;
};

//read result of pixel coordinate measurement of unknown point
void CResection::readfile_unknown(string filename) {
	int num_unknown;
	ifstream fin(filename);
	fin >> num_unknown;
	int id;
	PointPho tmp;
	for (int i = 0; i < num_unknown; i++) {
		fin >> id;
		id_unknown.push_back(id);

		fin >> tmp.x >> tmp.y;
		tmp.x -= COL / 2;//pixel
		tmp.y = ROW / 2 - tmp.y;//pixel
		tmp.x = tmp.x * pixel;//mm
		tmp.y = tmp.y * pixel;//mm
		left_unknown.push_back(tmp);

		fin >> tmp.x >> tmp.y;
		tmp.x -= COL / 2;//pixel
		tmp.y = ROW / 2 - tmp.y;//pixel
		tmp.x = tmp.x * pixel;//mm
		tmp.y = tmp.y * pixel;//mm
		right_unknown.push_back(tmp);

	}
	fin.close();
	cout << endl << "/******************************** Intersection ********************************/" << endl << endl;
	cout << "Read Unkown Photos num:" << num_unknown << endl;
};

//calculate rotation matrix
Mat CResection::Rotation(double omega, double phi, double kappa) {
	Mat matrix = Mat::zeros(3, 3, CV_64F);

	matrix.at<double>(0, 0) = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa);//a1
	matrix.at<double>(0, 1) = -cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa);//a2
	matrix.at<double>(0, 2) = -sin(phi) * cos(omega);//a3

	matrix.at<double>(1, 0) = cos(omega) * sin(kappa);//b1
	matrix.at<double>(1, 1) = cos(omega) * cos(kappa);//b2
	matrix.at<double>(1, 2) = -sin(omega);//b3

	matrix.at<double>(2, 0) = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa);//c1
	matrix.at<double>(2, 1) = -sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa);//c2
	matrix.at<double>(2, 2) = cos(phi) * cos(omega);//c3

	return matrix;
}

//calculate interior & exterior orientation elements & distortion coefficient
void CResection::cal_resection(string filename) {

	/*
	*	init:	V = A * t + C * X2 + D * Xad - L;
	*	|Vx| =	|a11 a12 a13 a14 a15 a16|[dXS;dYs;dZs;dphi;domega;dkappa] + |b11 b12 b13|[df;dx0;dy0] + |c11 c12 c13 c14|[dk1;dk2;dP1;dP2] - |x - (x)|
	*	|Vy|	|a21 a22 a23 a24 a25 a26|									|b21 b22 b23|				|c21 c22 c23 c24|					 |y - (y)|
	*			V(2n*1) = A(2n*13) * delta(13*1) - L(2n*1)
	*/

	Mat A(2 * num, 13, CV_64F), L(2 * num, 1, CV_64F), delta(13, 1, CV_64F), V = Mat::zeros(2 * num, 1, CV_64F);
	int flag = 0;  //iteration times

	cout << endl << "！！！！！！！！ interior & exterior orientation elements & distortion coefficient ！！！！！！！！" << endl << endl;

	do {
		//calculate rotation matrix
		double a1, a2, a3, b1, b2, b3, c1, c2, c3;
		a1 = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa);
		a2 = -cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa);
		a3 = -sin(phi) * cos(omega);
		b1 = cos(omega) * sin(kappa);
		b2 = cos(omega) * cos(kappa);
		b3 = -sin(omega);
		c1 = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa);
		c2 = -sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa);
		c3 = cos(phi) * cos(omega);

		//establish error equations point-by-point and normalize��V=A*x-L
		for (int i = 0; i < num; i++) {

			double x = photos[i].x, y = photos[i].y;
			double X, Y, Z;

			vector <int>::iterator iElement = find(id_gcps.begin(), id_gcps.end(), id_photos[i]);
			if (iElement != id_gcps.end())
			{
				int nPosition = distance(id_gcps.begin(), iElement);
				X = gcps[nPosition].X, Y = gcps[nPosition].Y, Z = gcps[nPosition].Z;
			}
			else
				continue;

			double rr = (x - x0) * (x - x0) + (y - y0) * (y - y0);
			double Xp = a1 * (X - Xs) + b1 * (Y - Ys) + c1 * (Z - Zs);
			double Yp = a2 * (X - Xs) + b2 * (Y - Ys) + c2 * (Z - Zs);
			double Zp = a3 * (X - Xs) + b3 * (Y - Ys) + c3 * (Z - Zs);

			A.at<double>(i * 2, 0) = 1.0 / Zp * (a1 * f + a3 * (x - x0));
			A.at<double>(i * 2, 1) = 1.0 / Zp * (b1 * f + b3 * (x - x0));
			A.at<double>(i * 2, 2) = 1.0 / Zp * (c1 * f + c3 * (x - x0));
			A.at<double>(i * 2, 3) = (y - y0) * sin(omega) - ((x - x0) / f * ((x - x0) * cos(kappa) - (y - y0) * sin(kappa)) + f * cos(kappa)) * cos(omega);
			A.at<double>(i * 2, 4) = -f * sin(kappa) - (x - x0) / f * ((x - x0) * sin(kappa) + (y - y0) * cos(kappa));
			A.at<double>(i * 2, 5) = y - y0;

			A.at<double>(i * 2, 6) = (x - x0) / f;
			A.at<double>(i * 2, 7) = 1;
			A.at<double>(i * 2, 8) = 0;

			A.at<double>(i * 2, 9) = -(x - x0) * rr;
			A.at<double>(i * 2, 10) = -(x - x0) * rr * rr;
			A.at<double>(i * 2, 11) = -rr - 2 * (x - x0) * (x - x0);
			A.at<double>(i * 2, 12) = -2 * (x - x0) * (y - y0);

			A.at<double>(i * 2 + 1, 0) = 1.0 / Zp * (a2 * f + a3 * (y - y0));
			A.at<double>(i * 2 + 1, 1) = 1.0 / Zp * (b2 * f + b3 * (y - y0));
			A.at<double>(i * 2 + 1, 2) = 1.0 / Zp * (c2 * f + c3 * (y - y0));
			A.at<double>(i * 2 + 1, 3) = -(x - x0) * sin(omega) - ((y - y0) / f * ((x - x0) * cos(kappa) - (y - y0) * sin(kappa)) - f * sin(kappa)) * cos(omega);
			A.at<double>(i * 2 + 1, 4) = -f * cos(kappa) - (y - y0) / f * ((x - x0) * sin(kappa) + (y - y0) * cos(kappa));
			A.at<double>(i * 2 + 1, 5) = -(x - x0);

			A.at<double>(i * 2 + 1, 6) = (y - y0) / f;
			A.at<double>(i * 2 + 1, 7) = 0;
			A.at<double>(i * 2 + 1, 8) = 1;

			A.at<double>(i * 2 + 1, 9) = -(y - y0) * rr;
			A.at<double>(i * 2 + 1, 10) = -(y - y0) * rr * rr;
			A.at<double>(i * 2 + 1, 11) = -2 * (x - x0) * (y - y0);
			A.at<double>(i * 2 + 1, 12) = -rr - 2 * (y - y0) * (y - y0);

			double dx = (x - x0) * (k1 * rr + k2 * rr * rr) + p1 * (rr + 2 * (x - x0) * (x - x0)) + 2 * p2 * (x - x0) * (y - y0),
				dy = (y - y0) * (k1 * rr + k2 * rr * rr) + p2 * (rr + 2 * (y - y0) * (y - y0)) + 2 * p1 * (x - x0) * (y - y0);

			L.at<double>(i * 2, 0) = x + f * Xp / Zp - x0 + dx;
			L.at<double>(i * 2 + 1, 0) = y + f * Yp / Zp - y0 + dy;

		}

		//d=(A'A)^-1，A'，L
		delta = (A.t() * A).inv() * (A.t() * L);//個屎方

		//correction result
		Xs += delta.at<double>(0, 0);
		Ys += delta.at<double>(1, 0);
		Zs += delta.at<double>(2, 0);
		phi += delta.at<double>(3, 0);
		omega += delta.at<double>(4, 0);
		kappa += delta.at<double>(5, 0);
		f += delta.at<double>(6, 0);
		x0 += delta.at<double>(7, 0);
		y0 += delta.at<double>(8, 0);
		k1 += delta.at<double>(9, 0);
		k2 += delta.at<double>(10, 0);
		p1 += delta.at<double>(11, 0);
		p2 += delta.at<double>(12, 0);

		flag++;

	} while (flag <= iter && abs(delta.at<double>(0, 0)) > 1e-15 && abs(delta.at<double>(1, 0)) > 1e-15 && abs(delta.at<double>(2, 0)) > 1e-15
		&& abs(delta.at<double>(3, 0)) > 1e-15 && abs(delta.at<double>(4, 0)) > 1e-15 && abs(delta.at<double>(5, 0)) > 1e-15
		&& abs(delta.at<double>(6, 0)) > 1e-15 && abs(delta.at<double>(7, 0)) > 1e-15 && abs(delta.at<double>(8, 0)) > 1e-15
		&& abs(delta.at<double>(9, 0)) > 1e-15 && abs(delta.at<double>(10, 0)) > 1e-15 && abs(delta.at<double>(11, 0)) > 1e-15 && abs(delta.at<double>(12, 0)) > 1e-15);

	//end
	if (flag > iter) {
		cout << "Error: overtime!";
		return;
	}
	else {
		//accuracy assessment
		Mat V = A * delta - L;
		Mat tmp = V.t() * V,
			Qvv = (A.t() * A).inv();
		double m0 = sqrt(tmp.at<double>(0, 0) / (2 * num - 13));//6+3+4=13

		//m0*Qxx[i]
		double m_Xs = m0 * sqrt(Qvv.at<double>(0, 0)),//Xs
			m_Ys = m0 * sqrt(Qvv.at<double>(1, 1)),//Ys
			m_Zs = m0 * sqrt(Qvv.at<double>(2, 2)),//Zs
			m_phi = m0 * sqrt(Qvv.at<double>(3, 3)),//phi
			m_omega = m0 * sqrt(Qvv.at<double>(4, 4)),//omega
			m_kappa = m0 * sqrt(Qvv.at<double>(5, 5)),//kappa
			m_f = m0 * sqrt(Qvv.at<double>(6, 6)),//f
			m_x0 = m0 * sqrt(Qvv.at<double>(7, 7)),//x0
			m_y0 = m0 * sqrt(Qvv.at<double>(8, 8)),//y0
			m_k1 = m0 * sqrt(Qvv.at<double>(9, 9)),//k1
			m_k2 = m0 * sqrt(Qvv.at<double>(10, 10)),//k2
			m_p1 = m0 * sqrt(Qvv.at<double>(11, 11)),//p1
			m_p2 = m0 * sqrt(Qvv.at<double>(12, 12));//p2

		//output
		cout << "iteration times: " << flag << endl << endl;
		cout << "m0 = \t" << setw(13) << m0 << "\t(mm)" << endl;
		cout << "Xs = \t" << -Zs << " \t(＼) " << setw(15) << m_Zs << "\t(mm)" << endl;
		cout << "Ys = \t" << Xs << " \t(＼) " << setw(15) << m_Xs << "\t(mm)" << endl;
		cout << "Zs = \t" << Ys << " \t(＼) " << setw(15) << m_Ys << "\t(mm)" << endl;
		cout << "phi = \t" << phi << " \t(＼) " << setw(15) << m_phi << "\t(rad)" << endl;
		cout << "omega =\t" << omega << " \t(＼) " << setw(15) << m_omega << "\t(rad)" << endl;
		cout << "kappa =\t" << kappa << " \t(＼) " << setw(15) << m_kappa << "\t(rad)" << endl;
		cout << "f = \t" << f << " \t(＼) " << setw(15) << m_f << "\t(mm)" << endl;
		cout << "x0 = \t" << x0 << " \t(＼) " << setw(15) << m_x0 << "\t(mm)" << endl;
		cout << "y0 = \t" << y0 << " \t(＼) " << setw(15) << m_y0 << "\t(mm)" << endl;
		cout << "k1 = \t" << k1 << " \t(＼) " << setw(15) << m_k1 << endl;
		cout << "k2 = \t" << k2 << " \t(＼) " << setw(15) << m_k2 << endl;
		cout << "p1 = \t" << p1 << " \t(＼) " << setw(15) << m_p1 << endl;
		cout << "p2 = \t" << p2 << " \t(＼) " << setw(15) << m_p2 << endl;

		ofstream foutput1("result/resction_" + filename + "(mm).txt");
		foutput1 << "iteration times: " << flag << endl << endl;
		foutput1 << "m0 = \t" << m0 << "\t(mm)" << endl;
		foutput1 << "Xs = \t" << setw(10) << -Zs << " \t(＼)" << m_Xs << "\t(mm)" << endl;
		foutput1 << "Ys = \t" << setw(10) << Xs << " \t(＼) " << m_Zs << "\t(mm)" << endl;
		foutput1 << "Zs = \t" << setw(10) << Ys << " \t(＼) " << m_Ys << "\t(mm)" << endl;
		foutput1 << "phi = \t" << setw(10) << phi << " \t(＼) " << m_phi << "\t(rad)" << endl;
		foutput1 << "omega =\t" << setw(10) << omega << " \t(＼) " << m_omega << "\t(rad)" << endl;
		foutput1 << "kappa =\t" << setw(10) << kappa << " \t(＼) " << m_kappa << "\t(rad)" << endl;
		foutput1 << "f = \t" << setw(10) << f << " \t(＼) " << m_f << "\t(mm)" << endl;
		foutput1 << "x0 = \t" << setw(10) << x0 << " \t(＼) " << m_x0 << "\t(mm)" << endl;
		foutput1 << "y0 = \t" << setw(10) << y0 << " \t(＼) " << m_y0 << "\t(mm)" << endl;
		foutput1 << "k1 = \t" << setw(10) << k1 << " \t(＼) " << m_k1 << endl;
		foutput1 << "k2 = \t" << setw(10) << k2 << " \t(＼) " << m_k2 << endl;
		foutput1 << "p1 = \t" << setw(10) << p1 << " \t(＼) " << m_p1 << endl;
		foutput1 << "p2 = \t" << setw(10) << p2 << " \t(＼) " << m_p2 << endl;
		foutput1.close();

		ofstream foutput2("result/points_" + filename + "(pixel).txt");
		for (int i = 0; i < num; i++) {
			foutput2 << id_photos[i] << "  " << setw(15) << V.at<double>(2 * i, 0) / pixel << "  " << setw(15) << V.at<double>(2 * i + 1, 0) / pixel << endl;
		}
		foutput2.close();

		return;
	}
}

//check point accuracy assessment for resection
//void CResction::outer_precision_check() {
//	double err_caliPoints_mean = 0;
//	int flag = 0;
//	cout << endl << "！！！！！！！！！！！ Check Point Accuracy Assessment for Resection (pixel) ！！！！！！！！！！！" << endl << endl;
//	cout << "                     x        y" << endl;
//	for (int i = num ; i < id_photos.size(); i++) {
//		double x = photos[i].x, y = photos[i].y;
//		double X, Y, Z;
//
//		vector <int>::iterator iElement = find(id_gcps.begin(), id_gcps.end(), id_photos[i]);
//		if (iElement != id_gcps.end())
//		{
//			int nPosition = distance(id_gcps.begin(), iElement);
//			X = gcps[nPosition].X, Y = gcps[nPosition].Y, Z = gcps[nPosition].Z;
//			flag++;
//		}
//		else
//			continue;
//
//		Mat R = Rotation(omega, phi, kappa);
//
//		double rr = (x - x0) * (x - x0) + (y - y0) * (y - y0);
//		x = x - x0 + ((x - x0) * (k1 * rr + k2 * rr * rr) + p1 * (rr + 2 * (x - x0) * (x - x0) + p2 * 2 * (x - x0) * (y - y0)));
//		y = y - y0 + ((y - y0) * (k1 * rr + k2 * rr * rr) + p1 * (rr + 2 * (y - y0) * (y - y0) + p2 * 2 * (x - x0) * (y - y0)));
//
//		double u = -f * (R.at<double>(0, 0) * (X - Xs) + R.at<double>(1, 0) * (Y - Ys) + R.at<double>(2, 0) * (Z - Zs)) / (R.at<double>(0, 2) * (X - Xs) + R.at<double>(1, 2) * (Y - Ys) + R.at<double>(2, 2) * (Z - Zs));
//		double v = -f * (R.at<double>(0, 1) * (X - Xs) + R.at<double>(1, 1) * (Y - Ys) + R.at<double>(2, 1) * (Z - Zs)) / (R.at<double>(0, 2) * (X - Xs) + R.at<double>(1, 2) * (Y - Ys) + R.at<double>(2, 2) * (Z - Zs));
//		double tmp = sqrt(pow(u - x, 2) + pow(v - y, 2));
//		cout << "Check Point " << id_photos[i] << " " << setw(8) << (u - x) / pixel << " " << setw(8) << (v - y) / pixel << "\tReprojection Error Accuracy : " << tmp / pixel << endl;
//
//		err_caliPoints_mean = err_caliPoints_mean + tmp;
//	}
//	cout << "Point Mean Error ��" << err_caliPoints_mean / (flag * pixel) << "\t" << endl;
//}

///DLT
CDLT::CDLT( string path_left, string path_right, string path_gcps, string path_unkown) {
	iops.x0 = 0;
	iops.y0 = 0;
	iops.fx = 18;//mm

	eops.Xs = 1500;//mm
	eops.Ys = 0;//mm
	eops.Zs = -1000;//mm
	eops.omega = 5 / 180.0 * PI;//rad
	eops.phi = 0 / 180.0 * PI;//rad
	eops.kappa = 0 / 180.0 * PI;//rad

	eops2.Xs = 3000;//mm
	eops2.Ys = 0;//mm
	eops2.Zs = -1007;//mm
	eops2.omega = -5 / 180.0 * PI;//rad
	eops2.phi = 0 / 180.0 * PI;//rad
	eops2.kappa = 0 / 180.0 * PI;//rad

	num = 50;

	L = Mat::zeros(15, 1, CV_64F);

	cout << endl << "/******************************** DLT ********************************/" << endl << endl;
	readfile_gcps(path_gcps);
	readfile_photos(path_left, path_right);
	readfile_unknown(path_unkown);


};
CDLT::~CDLT() {
	m.clear();
	id_photos_left.clear();
	id_photos_right.clear();
	id_gcps.clear();
	id_ctrl.clear();
	photos_left.clear();
	photos_right.clear();
	pho_ctrl.clear();
	gcps.clear();
	gcps_ctrl.clear();

	id_unknown.clear();
	left_unknown.clear();
	right_unknown.clear();
	gcps_unknown.clear();
};

//read ground coordinates of GCP (transform)
void CDLT::readfile_gcps(string filename) {
	ifstream fin(filename);
	int num_gcps, id, flag;
	PointGCP tmp;

	fin >> num_gcps;
	for (int i = 0; i < num_gcps; i++) {
		fin >> id >> tmp.Z >> tmp.X >> tmp.Y >> flag;
		tmp.Z = -tmp.Z;
		gcps.push_back(tmp);
		id_gcps.push_back(id);
	}
	fin.close();
	cout << "Read gcps num:" << num_gcps << endl;
};

//read result of pixel coordinate measurement
void CDLT::readfile_photos(string filename1, string filename2) {
	int num_left, num_right;

	ifstream fin(filename1);
	fin >> num_left;
	int id;
	PointPho tmp;
	for (int i = 0; i < num_left; i++) {
		fin >> id >> tmp.x >> tmp.y;
		tmp.x -= COL / 2;//mm
		tmp.y = ROW / 2 - tmp.y;//mm

		tmp.x = tmp.x * pixel;//mm
		tmp.y = tmp.y * pixel;//mm
		photos_left.push_back(tmp);
		id_photos_left.push_back(id);
	}
	fin.close();
	cout << "Read Left Photos num:" << num_left << endl;

	ifstream fin1(filename2);
	fin1 >> num_right;
	for (int i = 0; i < num_right; i++) {
		fin1 >> id >> tmp.x >> tmp.y;

		tmp.x -= COL / 2;//mm
		tmp.y = ROW / 2 - tmp.y;//mm

		tmp.x = tmp.x * pixel;//mm
		tmp.y = tmp.y * pixel;//mm

		photos_right.push_back(tmp);
		id_photos_right.push_back(id);
	}
	fin1.close();
	cout << "Read Right Photos num:" << num_right << endl;
};

//read result of pixel coordinate measurement (in pair)
void CDLT::readfile_unknown(string filename) {
	int num_unkown;
	ifstream fin(filename);
	fin >> num_unkown;
	int id;
	PointPho tmp;
	for (int i = 0; i < num_unkown; i++) {
		fin >> id;
		id_unknown.push_back(id);

		fin >> tmp.x >> tmp.y;
		tmp.x -= COL / 2;//mm
		tmp.y = ROW / 2 - tmp.y;//mm
		tmp.x = tmp.x * pixel;//mm
		tmp.y = tmp.y * pixel;//mm
		left_unknown.push_back(tmp);

		fin >> tmp.x >> tmp.y;
		tmp.x -= COL / 2;//mm
		tmp.y = ROW / 2 - tmp.y;//mm
		tmp.x = tmp.x * pixel;//mm
		tmp.y = tmp.y * pixel;//mm
		right_unknown.push_back(tmp);

	}
	fin.close();
	cout << "Read Unknown Photos num:" << num_unkown << endl;
};

//calculate rotation matrix
Mat CDLT::Rotation(double omega, double phi, double kappa) {
	Mat matrix = Mat::zeros(3, 3, CV_64F);

	matrix.at<double>(0, 0) = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa);//a1
	matrix.at<double>(0, 1) = -cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa);//a2
	matrix.at<double>(0, 2) = -sin(phi) * cos(omega);//a3

	matrix.at<double>(1, 0) = cos(omega) * sin(kappa);//b1
	matrix.at<double>(1, 1) = cos(omega) * cos(kappa);//b2
	matrix.at<double>(1, 2) = -sin(omega);//b3

	matrix.at<double>(2, 0) = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa);//c1
	matrix.at<double>(2, 1) = -sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa);//c2
	matrix.at<double>(2, 2) = cos(phi) * cos(omega);//c3

	return matrix;
}

//initialize control point
void CDLT::init_ctrl(vector<int> id, vector<PointPho> photos) {
	pho_ctrl = photos;
	for (int i = 0; i < num; i++) {
		id_ctrl.push_back(id[i]);
		vector <int>::iterator iElement = find(id_gcps.begin(), id_gcps.end(), id[i]);
		if (iElement != id_gcps.end())//get ground coordinate of control point
		{
			int nPosition = distance(id_gcps.begin(), iElement);
			PointGCP tmp;
			tmp.X = gcps[nPosition].X; tmp.Y = gcps[nPosition].Y; tmp.Z = gcps[nPosition].Z;
			gcps_ctrl.push_back(tmp);
		}
		else
			continue;
	}

}

//initialize L1~L11 and distortion coefficients (using first 6 control points)
Mat CDLT::init_L() {

	Mat A = Mat::zeros(2 * 6, 11, CV_64F), U = Mat::zeros(2 * 6, 1, CV_64F);
	Mat LL = Mat::zeros(15, 1, CV_64F);

	//establish error equations point-by-point and normalize
	// A*L=U
	for (int i = 0; i < 6; i++) {
		A.at<double>(2 * i, 0) = gcps_ctrl[i].X;
		A.at<double>(2 * i, 1) = gcps_ctrl[i].Y;
		A.at<double>(2 * i, 2) = gcps_ctrl[i].Z;
		A.at<double>(2 * i, 3) = 1;

		A.at<double>(2 * i, 8) = gcps_ctrl[i].X * pho_ctrl[i].x;
		A.at<double>(2 * i, 9) = gcps_ctrl[i].Y * pho_ctrl[i].x;
		A.at<double>(2 * i, 10) = gcps_ctrl[i].Z * pho_ctrl[i].x;

		A.at<double>(2 * i + 1, 4) = gcps_ctrl[i].X;
		A.at<double>(2 * i + 1, 5) = gcps_ctrl[i].Y;
		A.at<double>(2 * i + 1, 6) = gcps_ctrl[i].Z;
		A.at<double>(2 * i + 1, 7) = 1;

		A.at<double>(2 * i + 1, 8) = gcps_ctrl[i].X * pho_ctrl[i].y;
		A.at<double>(2 * i + 1, 9) = gcps_ctrl[i].Y * pho_ctrl[i].y;
		A.at<double>(2 * i + 1, 10) = gcps_ctrl[i].Z * pho_ctrl[i].y;

		U.at<double>(2 * i, 0) = -pho_ctrl[i].x;
		U.at<double>(2 * i + 1, 0) = -pho_ctrl[i].y;
	}
	Mat l = (A.t() * A).inv() * A.t() * U;
	for (int j = 0; j < 11; j++) {
		LL.at<double>(j, 0) = l.at<double>(j, 0);
	}
	for (int j = 0; j < 4; j++) {
		LL.at<double>(j + 11, 0) = 0;
	}
	return LL;
}

//get interior orientation parameters fx、fy
double CDLT::Getfx(Mat LL) {

	double garma = 1 / (LL.at<double>(8, 0) * LL.at<double>(8, 0) + LL.at<double>(9, 0) * LL.at<double>(9, 0) + LL.at<double>(10, 0) * LL.at<double>(10, 0));
	double tmpx0 = -(LL.at<double>(0, 0) * LL.at<double>(8, 0) + LL.at<double>(1, 0) * LL.at<double>(9, 0) + LL.at<double>(2, 0) * LL.at<double>(10, 0)) * garma;
	double tmpy0 = -(LL.at<double>(4, 0) * LL.at<double>(8, 0) + LL.at<double>(5, 0) * LL.at<double>(9, 0) + LL.at<double>(6, 0) * LL.at<double>(10, 0)) * garma;

	double A = (LL.at<double>(0, 0) * LL.at<double>(0, 0) + LL.at<double>(1, 0) * LL.at<double>(1, 0) + LL.at<double>(2, 0) * LL.at<double>(2, 0)) * garma - tmpx0 * tmpx0;
	double B = (LL.at<double>(4, 0) * LL.at<double>(4, 0) + LL.at<double>(5, 0) * LL.at<double>(5, 0) + LL.at<double>(6, 0) * LL.at<double>(6, 0)) * garma - tmpy0 * tmpy0;
	double C = (LL.at<double>(0, 0) * LL.at<double>(4, 0) + LL.at<double>(5, 0) * LL.at<double>(1, 0) + LL.at<double>(6, 0) * LL.at<double>(2, 0)) * garma - tmpx0 * tmpy0;

	double beta = sqrt(C * C / A / B);
	beta = C > 0 ? asin(-beta) : asin(beta);

	double fx = cos(beta) * sqrt(A);

	return fx;

}
double CDLT::Getfy(Mat LL) {

	double garma = 1 / (LL.at<double>(8, 0) * LL.at<double>(8, 0) + LL.at<double>(9, 0) * LL.at<double>(9, 0) + LL.at<double>(10, 0) * LL.at<double>(10, 0));
	double tmpx0 = -(LL.at<double>(0, 0) * LL.at<double>(8, 0) + LL.at<double>(1, 0) * LL.at<double>(9, 0) + LL.at<double>(2, 0) * LL.at<double>(10, 0)) * garma;
	double tmpy0 = -(LL.at<double>(4, 0) * LL.at<double>(8, 0) + LL.at<double>(5, 0) * LL.at<double>(9, 0) + LL.at<double>(6, 0) * LL.at<double>(10, 0)) * garma;

	double A = (LL.at<double>(0, 0) * LL.at<double>(0, 0) + LL.at<double>(1, 0) * LL.at<double>(1, 0) + LL.at<double>(2, 0) * LL.at<double>(2, 0)) * garma - tmpx0 * tmpx0;
	double B = (LL.at<double>(4, 0) * LL.at<double>(4, 0) + LL.at<double>(5, 0) * LL.at<double>(5, 0) + LL.at<double>(6, 0) * LL.at<double>(6, 0)) * garma - tmpy0 * tmpy0;
	double C = (LL.at<double>(0, 0) * LL.at<double>(4, 0) + LL.at<double>(5, 0) * LL.at<double>(1, 0) + LL.at<double>(6, 0) * LL.at<double>(2, 0)) * garma - tmpx0 * tmpy0;

	double fy = sqrt((A * B - C * C) / A);

	return fy;

}

//caculate L1~L11 and distortion coefficients
Mat CDLT::cal_L(vector<int> id, vector<PointPho> photos, string filename) {

	pho_ctrl.clear(); 
	id_ctrl.clear(); 
	gcps_ctrl.clear();

	//initialize
	init_ctrl(id, photos);
	Mat LL = init_L();

	int flag = 0;
	int size = id_ctrl.size();
	Mat M = Mat::zeros(2 * size, 15, CV_64F), W = Mat::zeros(2 * size, 1, CV_64F);
	double before = 0, now = 0;
	cout << endl << "！！！！！！！！ Solving for DLT Parameters！！！！！！！！" << endl << endl;
	cout << endl << "Control Points Num:" << size << endl;

	do {
		before = Getfx(LL);

		//establish error equations point-by-point and normalize
		for (int i = 0; i < size; i++) {
			double A = LL.at<double>(8, 0) * gcps_ctrl[i].X + LL.at<double>(9, 0) * gcps_ctrl[i].Y + LL.at<double>(10, 0) * gcps_ctrl[i].Z + 1;
			double x0 = -(LL.at<double>(0, 0) * LL.at<double>(8, 0) + LL.at<double>(1, 0) * LL.at<double>(9, 0) + LL.at<double>(2, 0) * LL.at<double>(10, 0)) / (LL.at<double>(8, 0) * LL.at<double>(8, 0) + LL.at<double>(9, 0) * LL.at<double>(9, 0) + LL.at<double>(10, 0) * LL.at<double>(10, 0));
			double y0 = -(LL.at<double>(4, 0) * LL.at<double>(8, 0) + LL.at<double>(5, 0) * LL.at<double>(9, 0) + LL.at<double>(6, 0) * LL.at<double>(10, 0)) / (LL.at<double>(8, 0) * LL.at<double>(8, 0) + LL.at<double>(9, 0) * LL.at<double>(9, 0) + LL.at<double>(10, 0) * LL.at<double>(10, 0));

			double r = sqrt(pow(pho_ctrl[i].x - x0, 2) + pow(pho_ctrl[i].y - y0, 2));

			M.at<double>(2 * i, 0) = -gcps_ctrl[i].X / A;
			M.at<double>(2 * i, 1) = -gcps_ctrl[i].Y / A;
			M.at<double>(2 * i, 2) = -gcps_ctrl[i].Z / A;
			M.at<double>(2 * i, 3) = -1.0 / A;
			M.at<double>(2 * i, 8) = -gcps_ctrl[i].X * pho_ctrl[i].x / A;
			M.at<double>(2 * i, 9) = -gcps_ctrl[i].Y * pho_ctrl[i].x / A;
			M.at<double>(2 * i, 10) = -gcps_ctrl[i].Z * pho_ctrl[i].x / A;

			M.at<double>(2 * i + 1, 4) = -gcps_ctrl[i].X / A;
			M.at<double>(2 * i + 1, 5) = -gcps_ctrl[i].Y / A;
			M.at<double>(2 * i + 1, 6) = -gcps_ctrl[i].Z / A;
			M.at<double>(2 * i + 1, 7) = -1.0 / A;
			M.at<double>(2 * i + 1, 8) = -gcps_ctrl[i].X * pho_ctrl[i].y / A;
			M.at<double>(2 * i + 1, 9) = -gcps_ctrl[i].Y * pho_ctrl[i].y / A;
			M.at<double>(2 * i + 1, 10) = -gcps_ctrl[i].Z * pho_ctrl[i].y / A;

			M.at<double>(2 * i, 11) = -(pho_ctrl[i].x - x0) * pow(r, 2);
			M.at<double>(2 * i, 12) = -(pho_ctrl[i].x - x0) * pow(r, 4);
			M.at<double>(2 * i, 13) = -(2 * pow(pho_ctrl[i].x - x0, 2) + pow(r, 2));
			M.at<double>(2 * i, 14) = -(2 * (pho_ctrl[i].x - x0) * (pho_ctrl[i].y - y0));

			M.at<double>(2 * i + 1, 11) = -(pho_ctrl[i].y - y0) * pow(r, 2);
			M.at<double>(2 * i + 1, 12) = -(pho_ctrl[i].y - y0) * pow(r, 4);
			M.at<double>(2 * i + 1, 13) = -(2 * (pho_ctrl[i].x - x0) * (pho_ctrl[i].y - y0));
			M.at<double>(2 * i + 1, 14) = -(2 * pow(pho_ctrl[i].y - y0, 2) + pow(r, 2));

			W.at<double>(2 * i, 0) = pho_ctrl[i].x / A;
			W.at<double>(2 * i + 1, 0) = pho_ctrl[i].y / A;
		}
		LL = (M.t() * M).inv() * M.t() * W;

		flag++;

		now = Getfx(LL);

	} while (flag <= iter && abs(before - now) > 1e-2);

	//Accuracy assessment
	Mat V = M * LL - W;//(2n*15)(15*1)-(2n*1)
	Mat tmp = V.t() * V,
		Qvv = (M.t() * M).inv();
	double m0 = sqrt(tmp.at<double>(0, 0) / (2 * num - 15));//11+4=15
	//calculate mean error for each parameter
	for (int i = 0; i < 15; i++) {
		m.push_back(m0 * sqrt(Qvv.at<double>(i, i)));
	}

	//output
	cout << "iteration times: " << flag << endl << endl;
	cout << "L1~L11: " << endl;
	for (int i = 0; i < 11; i++) {
		cout << LL.at<double>(i, 0) << endl;
	}
	cout << endl;
	cout << "Distortion Coefficients: (k1 k2 p1 p2)" << endl;
	for (int i = 11; i < 15; i++) {
		cout << LL.at<double>(i, 0) << endl;
	}
	cout << endl;
	cout << "Point Mean Error : " << m0 << endl;
	cout << "Mean Error of Parameters : " << endl;
	for (int i = 0; i < 15; i++) {
		cout << " " << m[i] << endl;
	}
	cout << endl;

	cout << "Residuals of Image Point : (pixel)" << m0 << endl;
	cout << "                     Vx               Vy" << endl;
	ofstream foutput("result/DLT_point_" + filename + "(pixel).txt");
	if (filename == "left") {
		for (int i = 0; i < num; i++) {
			foutput << id_photos_left[i] << "  " << setw(15) << V.at<double>(2 * i, 0) / pixel << "  " << setw(15) << V.at<double>(2 * i + 1, 0) / pixel << endl;
			cout << "Point " << id_photos_left[i] << "  " << setw(15) << V.at<double>(2 * i, 0) / pixel << "  " << setw(15) << V.at<double>(2 * i + 1, 0) / pixel << endl;
		}
	}
	else
	{
		for (int i = 0; i < num; i++) {
			foutput << id_photos_right[i] << "  " << setw(15) << V.at<double>(2 * i, 0) / pixel << "  " << setw(15) << V.at<double>(2 * i + 1, 0) / pixel << endl;
			cout << "Point " << id_photos_right[i] << "  " << setw(15) << V.at<double>(2 * i, 0) / pixel << "  " << setw(15) << V.at<double>(2 * i + 1, 0) / pixel << endl;
		}
	}
	
	foutput.close();

	return LL;
}

//estimation of interior and exterior orientation parameters
IOP CDLT::GetIOP(Mat LL) {
	IOP iop;

	double garma = 1 / (LL.at<double>(8, 0) * LL.at<double>(8, 0) + LL.at<double>(9, 0) * LL.at<double>(9, 0) + LL.at<double>(10, 0) * LL.at<double>(10, 0));
	iop.x0 = -(LL.at<double>(0, 0) * LL.at<double>(8, 0) + LL.at<double>(1, 0) * LL.at<double>(9, 0) + LL.at<double>(2, 0) * LL.at<double>(10, 0)) * garma;
	iop.y0 = -(LL.at<double>(4, 0) * LL.at<double>(8, 0) + LL.at<double>(5, 0) * LL.at<double>(9, 0) + LL.at<double>(6, 0) * LL.at<double>(10, 0)) * garma;

	double A = (LL.at<double>(0, 0) * LL.at<double>(0, 0) + LL.at<double>(1, 0) * LL.at<double>(1, 0) + LL.at<double>(2, 0) * LL.at<double>(2, 0)) * garma - iop.x0 * iop.x0;
	double B = (LL.at<double>(4, 0) * LL.at<double>(4, 0) + LL.at<double>(5, 0) * LL.at<double>(5, 0) + LL.at<double>(6, 0) * LL.at<double>(6, 0)) * garma - iop.y0 * iop.y0;
	double C = (LL.at<double>(0, 0) * LL.at<double>(4, 0) + LL.at<double>(5, 0) * LL.at<double>(1, 0) + LL.at<double>(6, 0) * LL.at<double>(2, 0)) * garma - iop.x0 * iop.y0;

	double dbeta = sqrt(C * C / A / B);
	dbeta = C > 0 ? asin(-dbeta) : asin(dbeta);

	double ds = sqrt(A / B) - 1;

	iop.fx = Getfx(LL);
	iop.fy = Getfy(LL);
	iop.dbeta = dbeta;
	iop.ds = ds;

	return iop;
}
EOP CDLT::GetEOP(Mat LL) {
	EOP eop;

	Mat AA = Mat::zeros(3, 3, CV_64F), bb = Mat::zeros(3, 1, CV_64F), XX = Mat::zeros(3, 1, CV_64F);
	AA.at<double>(0, 0) = LL.at<double>(0, 0);
	AA.at<double>(0, 1) = LL.at<double>(1, 0);
	AA.at<double>(0, 2) = LL.at<double>(2, 0);
	bb.at<double>(0, 0) = -LL.at<double>(3, 0);

	AA.at<double>(1, 0) = LL.at<double>(4, 0);
	AA.at<double>(1, 1) = LL.at<double>(5, 0);
	AA.at<double>(1, 2) = LL.at<double>(6, 0);
	bb.at<double>(1, 0) = -LL.at<double>(7, 0);

	AA.at<double>(2, 0) = LL.at<double>(8, 0);
	AA.at<double>(2, 1) = LL.at<double>(9, 0);
	AA.at<double>(2, 2) = LL.at<double>(10, 0);
	bb.at<double>(2, 0) = -1;

	Mat SS, UU, Vt;

	SVD::compute(AA, SS, UU, Vt, SVD::FULL_UV);

	XX = AA.inv() * bb;

	eop.Xs = XX.at<double>(0, 0);
	eop.Ys = XX.at<double>(1, 0);
	eop.Zs = XX.at<double>(2, 0);

	double garma = 1 / (LL.at<double>(8, 0) * LL.at<double>(8, 0) + LL.at<double>(9, 0) * LL.at<double>(9, 0) + LL.at<double>(10, 0) * LL.at<double>(10, 0));
	double a3 = LL.at<double>(8, 0) * sqrt(garma);
	double b3 = LL.at<double>(9, 0) * sqrt(garma);
	double c3 = LL.at<double>(10, 0) * sqrt(garma);
	double a2 = (sqrt(garma) * LL.at<double>(4, 0) + a3 * sqrt(garma)) * (1 + iops.ds) * cos(iops.dbeta) / iops.fx;
	double b2 = (LL.at<double>(5, 0) * sqrt(garma) + b3 * iops.y0) * (1 + iops.ds) * cos(iops.dbeta) / iops.fx;
	double b1 = ((LL.at<double>(1, 0) * sqrt(garma)) + b3 * iops.x0 + b2 * iops.fx * tan(iops.dbeta)) / iops.fx;

	eop.phi = atan(-a3 / c3);
	eop.omega = asin(-b3);
	eop.kappa = atan(b1 / b2);

	return eop;
}

//get DLT parameters
//calxulate L matrix, output distortion coefficients and interior & exterior orientation parameters
void CDLT::GetElement(vector<int> id, vector<PointPho> photos, string tag) {

	if (tag == "left") {
		L = cal_L(id, photos, tag);

		iops = GetIOP(L);
		eops = GetEOP(L);

		cout << endl << "！！！！！ interior & exterior orientation parameters and distortion coefficients ！！！！！" << endl << endl;

		cout << "Xs = \t" << -eops.Zs << "\t(mm)" << endl;
		cout << "Ys = \t" << eops.Xs << "\t(mm)" << endl;
		cout << "Zs = \t" << eops.Ys << "\t(mm)" << endl;
		cout << "phi = \t" << eops.phi << "\t(rad)" << endl;
		cout << "omega =\t" << eops.omega << "\t(rad)" << endl;
		cout << "kappa =\t" << eops.kappa << "\t(rad)" << endl;

		cout << "x0 = \t" << iops.x0 << "\t(mm)" << endl;
		cout << "y0 = \t" << iops.y0 << "\t(mm)" << endl;
		cout << "fx = \t" << iops.fx << "\t(mm)" << endl;
		cout << "fy= \t" << iops.fy << "\t(mm)" << endl;
		cout << "dbeta = \t" << iops.dbeta << "\t(mm)" << endl;
		cout << "ds= \t" << iops.ds << "\t(mm)" << endl;

		cout << "k1 = \t" << L.at<double>(11, 0) << endl;
		cout << "k2 = \t" << L.at<double>(12, 0) << endl;
		cout << "p1 = \t" << L.at<double>(13, 0) << endl;
		cout << "p2 = \t" << L.at<double>(14, 0) << endl;

		ofstream foutput("result/DLT_" + tag + "(mm).txt");
		foutput << "L1~L11: \t";
		for (int i = 0; i < 11; i++) {
			foutput << setw(10) << L.at<double>(i, 0) << "  ";
		}
		foutput << "Distortion Coefficients: (k1 k2 p1 p2) \t";
		for (int i = 11; i < 15; i++) {
			foutput << setw(10) << L.at<double>(i, 0) << "  ";
		}
		foutput << endl << "Xs = \t" << -eops.Zs << "\t(mm)" << endl;
		foutput << "Ys = \t" << eops.Xs << "\t(mm)" << endl;
		foutput << "Zs = \t" << eops.Ys << "\t(mm)" << endl;
		foutput << "phi = \t" << eops.phi << "\t(rad)" << endl;
		foutput << "omega =\t" << eops.omega << "\t(rad)" << endl;
		foutput << "kappa =\t" << eops.kappa << "\t(rad)" << endl;

		foutput << "x0 = \t" << iops.x0 << "\t(mm)" << endl;
		foutput << "y0 = \t" << iops.y0 << "\t(mm)" << endl;
		foutput << "fx = \t" << iops.fx << "\t(mm)" << endl;
		foutput << "fy= \t" << iops.fy << "\t(mm)" << endl;
		foutput << "dbeta = \t" << iops.dbeta << "\t(mm)" << endl;
		foutput << "ds= \t" << iops.ds << "\t(mm)" << endl;

		foutput << "k1 = \t" << L.at<double>(11, 0) << endl;
		foutput << "k2 = \t" << L.at<double>(12, 0) << endl;
		foutput << "p1 = \t" << L.at<double>(13, 0) << endl;
		foutput << "p2 = \t" << L.at<double>(14, 0) << endl;
		foutput.close();
	}
	else if (tag == "right") {
		L2 = cal_L(id, photos, tag);

		iops2 = GetIOP(L2);
		eops2 = GetEOP(L2);

		cout << endl << "！！！！！ interior & exterior orientation parameters and distortion coefficients ！！！！！" << endl << endl;

		cout << "Xs = \t" << -eops2.Zs << "\t(mm)" << endl;
		cout << "Ys = \t" << eops2.Xs << "\t(mm)" << endl;
		cout << "Zs = \t" << eops2.Ys << "\t(mm)" << endl;
		cout << "phi = \t" << eops2.phi << "\t(rad)" << endl;
		cout << "omega =\t" << eops2.omega << "\t(rad)" << endl;
		cout << "kappa =\t" << eops2.kappa << "\t(rad)" << endl;

		cout << "x0 = \t" << iops2.x0 << "\t(mm)" << endl;
		cout << "y0 = \t" << iops2.y0 << "\t(mm)" << endl;
		cout << "fx = \t" << iops2.fx << "\t(mm)" << endl;
		cout << "fy= \t" << iops2.fy << "\t(mm)" << endl;
		cout << "dbeta = \t" << iops2.dbeta << "\t(mm)" << endl;
		cout << "ds= \t" << iops2.ds << "\t(mm)" << endl;

		cout << "k1 = \t" << L.at<double>(11, 0) << endl;
		cout << "k2 = \t" << L.at<double>(12, 0) << endl;
		cout << "p1 = \t" << L.at<double>(13, 0) << endl;
		cout << "p2 = \t" << L.at<double>(14, 0) << endl;

		ofstream foutput("result/DLT_" + tag + "(mm).txt");
		foutput << "L1~L11: \t";
		for (int i = 0; i < 15; i++) {
			foutput << setw(10) << L.at<double>(i, 0) << "  ";
		}
		foutput << endl << "Xs = \t" << -eops2.Zs << "\t(mm)" << endl;
		foutput << "Ys = \t" << eops2.Xs << "\t(mm)" << endl;
		foutput << "Zs = \t" << eops2.Ys << "\t(mm)" << endl;
		foutput << "phi = \t" << eops2.phi << "\t(rad)" << endl;
		foutput << "omega =\t" << eops2.omega << "\t(rad)" << endl;
		foutput << "kappa =\t" << eops2.kappa << "\t(rad)" << endl;

		foutput << "x0 = \t" << iops2.x0 << "\t(mm)" << endl;
		foutput << "y0 = \t" << iops2.y0 << "\t(mm)" << endl;
		foutput << "fx = \t" << iops2.fx << "\t(mm)" << endl;
		foutput << "fy= \t" << iops2.fy << "\t(mm)" << endl;
		foutput << "dbeta = \t" << iops2.dbeta << "\t(mm)" << endl;
		foutput << "ds= \t" << iops2.ds << "\t(mm)" << endl;

		foutput << "k1 = \t" << L.at<double>(11, 0) << endl;
		foutput << "k2 = \t" << L.at<double>(12, 0) << endl;
		foutput << "p1 = \t" << L.at<double>(13, 0) << endl;
		foutput << "p2 = \t" << L.at<double>(14, 0) << endl;
		foutput.close();
	}


}

//check point accuracy assessment for DLT
//void CDLT::outer_precision_check(vector<int> id_photos, vector<PointPho> photos) {
//
//	cout << endl << "！！！！！！！！！ Check Point Accuracy Assessment for DLT (pixel) ！！！！！！！！！" << endl;
//
//	double Var_x = 0, Var_y = 0;
//	cout << endl << "Check Points " << id_photos.size() - num << endl;
//
//	for (int i = num; i < id_photos.size(); i++) {
//		double x = photos[i].x, y = photos[i].y;
//		double X, Y, Z;
//
//		vector <int>::iterator iElement = find(id_gcps.begin(), id_gcps.end(), id_photos[i]);
//		if (iElement != id_gcps.end())
//		{
//			int nPosition = distance(id_gcps.begin(), iElement);
//			X = gcps[nPosition].X, Y = gcps[nPosition].Y, Z = gcps[nPosition].Z;
//		}
//		else
//			continue;
//
//		double rr = (x - iops.x0) * (x - iops.x0) + (y - iops.y0) * (y - iops.y0);
//		x = x + ((x - iops.x0) * (L.at<double>(11, 0) * rr + L.at<double>(12, 0) * rr * rr) + L.at<double>(13, 0) * (rr + 2 * (x - iops.x0) * (x - iops.x0) + L.at<double>(14, 0) * 2 * (x - iops.x0) * (y - iops.y0)));
//		y = y + ((y - iops.y0) * (L.at<double>(11, 0) * rr + L.at<double>(12, 0) * rr * rr) + L.at<double>(13, 0) * (rr + 2 * (y - iops.y0) * (y - iops.y0) + L.at<double>(14, 0) * 2 * (x - iops.x0) * (y - iops.y0)));
//
//		double vx = -(X * L.at<double>(0, 0) + Y * L.at<double>(1, 0) + Z * L.at<double>(2, 0) + L.at<double>(3, 0)) / (X * L.at<double>(8, 0) + Y * L.at<double>(9, 0) + Z * L.at<double>(10, 0) + 1) - x;
//		double vy = -(X * L.at<double>(4, 0) + Y * L.at<double>(5, 0) + Z * L.at<double>(6, 0) + L.at<double>(7, 0)) / (X * L.at<double>(8, 0) + Y * L.at<double>(9, 0) + Z * L.at<double>(10, 0) + 1) - y;
//
//		Var_x += abs(vx);
//		Var_y += abs(vy);
//
//		cout << "Check Point " << id_photos[i] << " " << vx / 0.00519663 << " " << vy / 0.00519663 << endl;
//
//	}
//
//	Var_x /= id_photos.size() - num;
//	Var_y /= id_photos.size() - num;
//
//	cout << endl << "Reprojection Error Accuracy��" << endl
//		<< " x= " << double(Var_x / (pixel)) << " pixel" << endl
//		<< " y= " << double(Var_y / (pixel)) << " pixel" << endl;
//}

//initialize XYZ
void CDLT::init_XYZ() {

	Mat A = Mat::zeros(3, 3, CV_64F), U = Mat::zeros(3, 1, CV_64F), XYZ;

	for (int i = 0; i < id_unknown.size(); i++) {

		A.at<double>(0, 0) = L.at<double>(0, 0) + left_unknown[i].x * L.at<double>(8, 0);
		A.at<double>(0, 1) = L.at<double>(1, 0) + left_unknown[i].x * L.at<double>(9, 0);
		A.at<double>(0, 2) = L.at<double>(2, 0) + left_unknown[i].x * L.at<double>(10, 0);

		A.at<double>(1, 0) = L.at<double>(4, 0) + left_unknown[i].y * L.at<double>(8, 0);
		A.at<double>(1, 1) = L.at<double>(5, 0) + left_unknown[i].y * L.at<double>(9, 0);
		A.at<double>(1, 2) = L.at<double>(6, 0) + left_unknown[i].y * L.at<double>(10, 0);

		A.at<double>(2, 0) = L2.at<double>(0, 0) + right_unknown[i].y * L2.at<double>(8, 0);
		A.at<double>(2, 1) = L2.at<double>(1, 0) + right_unknown[i].y * L2.at<double>(9, 0);
		A.at<double>(2, 2) = L2.at<double>(2, 0) + right_unknown[i].y * L2.at<double>(10, 0);

		U.at<double>(0, 0) = L.at<double>(3, 0) + left_unknown[i].x;
		U.at<double>(1, 0) = L.at<double>(7, 0) + left_unknown[i].y;
		U.at<double>(2, 0) = L2.at<double>(3, 0) + right_unknown[i].x;

		XYZ = A.inv() * U;

		PointGCP tmp;
		tmp.X = XYZ.at<double>(0, 0), tmp.Y = XYZ.at<double>(1, 0), tmp.Z = XYZ.at<double>(2, 0);

		gcps_unknown.push_back(tmp);

	}
}

//calculate XYZ
void CDLT::cal_XYZ() {

	int size = gcps_unknown.size();
	Mat N = Mat::zeros(4, 3, CV_64F), S = Mat::zeros(3, 1, CV_64F), Q = Mat::zeros(4, 1, CV_64F);

	cout << endl << "！！！！！！！！！！！ Coordinates of Unknown Points ！！！！！！！！！！！" << endl << endl;
	cout << "Unkown Points Num:" << size << endl;

	ofstream foutput("result/DLT_XYZ.txt");
	double X, Y, Z;
	//establish error equations point-by-point and normalize
	for (int i = 0; i < size; i++) {

		int flag = 0;
		Mat after = Mat::zeros(3, 1, CV_64F);

		double x1 = left_unknown[i].x, y1 = left_unknown[i].y, x2 = right_unknown[i].x, y2 = right_unknown[i].y;

		double rr = (x1 - iops.x0) * (x1 - iops.x0) + (y1 - iops.y0) * (y1 - iops.y0);
		x1 += ((x1 - iops.x0) * (L.at<double>(11, 0) * rr + L.at<double>(12, 0) * rr * rr) + L.at<double>(13, 0) * (rr + 2 * (x1 - iops.x0) * (x1 - iops.x0) + L.at<double>(14, 0) * 2 * (x1 - iops.x0) * (y1 - iops.y0)));
		y1 += ((y1 - iops.y0) * (L.at<double>(11, 0) * rr + L.at<double>(12, 0) * rr * rr) + L.at<double>(13, 0) * (rr + 2 * (y1 - iops.y0) * (y1 - iops.y0) + L.at<double>(14, 0) * 2 * (x1 - iops.x0) * (y1 - iops.y0)));

		rr = (x2 - iops.x0) * (x2 - iops.x0) + (y2 - iops.y0) * (y2 - iops.y0);
		x2 += ((x2 - iops.x0) * (L.at<double>(11, 0) * rr + L.at<double>(12, 0) * rr * rr) + L.at<double>(13, 0) * (rr + 2 * (x2 - iops.x0) * (x2 - iops.x0) + L.at<double>(14, 0) * 2 * (x2 - iops.x0) * (y2 - iops.y0)));
		y2 += ((y2 - iops.y0) * (L.at<double>(11, 0) * rr + L.at<double>(12, 0) * rr * rr) + L.at<double>(13, 0) * (rr + 2 * (y2 - iops.y0) * (y2 - iops.y0) + L.at<double>(14, 0) * 2 * (x2 - iops.x0) * (y2 - iops.y0)));

		after.at<double>(0, 0) = gcps_unknown[i].X;
		after.at<double>(1, 0) = gcps_unknown[i].Y;
		after.at<double>(2, 0) = gcps_unknown[i].Z;

		do {

			double X = after.at<double>(0, 0);
			double Y = after.at<double>(1, 0);
			double Z = after.at<double>(2, 0);

			double A = L.at<double>(8, 0) * gcps_ctrl[i].X + L.at<double>(9, 0) * gcps_ctrl[i].Y + L.at<double>(10, 0) * gcps_ctrl[i].Z + 1;
			double A2 = L2.at<double>(8, 0) * gcps_ctrl[i].X + L2.at<double>(9, 0) * gcps_ctrl[i].Y + L2.at<double>(10, 0) * gcps_ctrl[i].Z + 1;

			N.at<double>(0, 0) = -(L.at<double>(0, 0) + L.at<double>(8, 0) * x1) / A;
			N.at<double>(0, 1) = -(L.at<double>(1, 0) + L.at<double>(9, 0) * x1) / A;
			N.at<double>(0, 2) = -(L.at<double>(2, 0) + L.at<double>(10, 0) * x1) / A;

			N.at<double>(1, 0) = -(L.at<double>(4, 0) + L.at<double>(8, 0) * y1) / A;
			N.at<double>(1, 1) = -(L.at<double>(5, 0) + L.at<double>(9, 0) * y1) / A;
			N.at<double>(1, 2) = -(L.at<double>(6, 0) + L.at<double>(10, 0) * y1) / A;

			N.at<double>(2, 0) = -(L2.at<double>(0, 0) + L2.at<double>(8, 0) * x2) / A2;
			N.at<double>(2, 1) = -(L2.at<double>(1, 0) + L2.at<double>(9, 0) * x2) / A2;
			N.at<double>(2, 2) = -(L2.at<double>(2, 0) + L2.at<double>(10, 0) * x2) / A2;

			N.at<double>(3, 0) = -(L2.at<double>(4, 0) + L2.at<double>(8, 0) * y2) / A2;
			N.at<double>(3, 1) = -(L2.at<double>(5, 0) + L2.at<double>(9, 0) * y2) / A2;
			N.at<double>(3, 2) = -(L2.at<double>(6, 0) + L2.at<double>(10, 0) * y2) / A2;

			Q.at<double>(0, 0) = (L.at<double>(3, 0) + x1) / A;
			Q.at<double>(1, 0) = (L.at<double>(7, 0) + y1) / A;

			Q.at<double>(2, 0) = (L2.at<double>(3, 0) + x2) / A2;
			Q.at<double>(3, 0) = (L2.at<double>(7, 0) + y2) / A2;

			after = (N.t() * N).inv() * N.t() * Q;

			flag++;

			if (abs(X - after.at<double>(0, 0)) < 1e-2 && abs(Y - after.at<double>(1, 0)) < 1e-2 && abs(Z - after.at<double>(2, 0)) < 1e-2)
			{
				S.at<double>(0, 0) = after.at<double>(0, 0);
				S.at<double>(1, 0) = after.at<double>(1, 0);
				S.at<double>(2, 0) = after.at<double>(2, 0);
				break;
			}

		} while (flag <= iter);

		gcps_unknown[i].X = -S.at<double>(2, 0);
		gcps_unknown[i].Y = S.at<double>(0, 0);
		gcps_unknown[i].Z = S.at<double>(1, 0);

		cout << "Unkown " << id_unknown[i] << "  " << gcps_unknown[i].X << "  " << gcps_unknown[i].Y << "  " << gcps_unknown[i].Z << endl;
		foutput << id_unknown[i] << " " << gcps_unknown[i].X << "  " << gcps_unknown[i].Y << "  " << gcps_unknown[i].Z << endl;

		if (id_unknown[i] == 52) {
			X= gcps_unknown[i].X, Y= gcps_unknown[i].Y, Z= gcps_unknown[i].Z;
		}
	}

	cout << endl << "！！！！！！！！！！！Distances From Point 52 (mm) ！！！！！！！！！！！" << endl << endl;
	for (int i = 0; i < size; i++) {
		if (id_unknown[i]<100) {
			cout << "Point " << id_unknown[i] << "  " << sqrt(pow(gcps_unknown[i].X - X, 2) + pow(gcps_unknown[i].Y - Y, 2) + pow(gcps_unknown[i].Z - Z, 2)) << endl;
		}
	}

}

void CDLT::XYZ_precision_check() {
	double VX, VY, VZ, V = 0;
	int flag = 0;
	cout << endl;
	ofstream fout("result/DLT_PointCheck(mm).txt");
	cout << "！！！！！！！！！！！！！ Residuals of Check Point Coordinates (mm) ！！！！！！！！！！！！！" << endl << endl;
	for (int i = 0; i < id_unknown.size(); i++) {
		vector <int>::iterator iElement = find(id_gcps.begin(), id_gcps.end(), id_unknown[i]);
		if (iElement != id_gcps.end())
		{
			int nPosition = distance(id_gcps.begin(), iElement);

			PointGCP tmp = gcps[nPosition];

			VX = -tmp.Z - gcps_unknown[i].X; VY = tmp.X - gcps_unknown[i].Y; VZ = tmp.Y - gcps_unknown[i].Z;

			cout << id_unknown[i] << " " << VX << " " << VY << " " << VZ << endl;
			fout << id_unknown[i] << " " << VX << " " << VY << " " << VZ << endl;

			V += sqrt(VX * VX + VY * VY + VZ * VZ);
			flag++;

		}
		else
			continue;
	}
	cout << endl << "Check Point Num:" << flag << endl;
	cout << "Point Error Means:" << V / flag << "\t(mm)" << endl;
}

//resection_intersection
void Re_Intersection() {

	string path_gcps = "data/GCP.txt";
	string path_left = "data/left.txt";
	string path_right = "data/right.txt";
	string path_unknown = "data/pair_unknown.txt";

	IOP iops1, iops2;

	iops1.x0 = 0.16;
	iops1.y0 = 0.07;

	iops2.x0 = 0.08;
	iops2.y0 = 0.07;

	EOP eops1, eops2;

	eops1.Xs = 1500;
	eops1.Ys = 0;
	eops1.Zs = -1000;
	eops1.omega = 10 / 180.0 * PI;
	eops1.phi = 1.6 / 180.0 * PI;
	eops1.kappa = 0 / 180.0 * PI;

	eops2.Xs = 3000;
	eops2.Ys = 0;
	eops2.Zs = -1000;
	eops2.omega = -10 / 180.0 * PI;
	eops2.phi = -1 / 180.0 * PI;
	eops2.kappa = 0 / 180.0 * PI;

	cout << endl << "/******************************** Resection ********************************/" << endl << endl;
	cout << endl << "！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！" << endl;
	cout << "Left Image processing..." << endl;
	CResection rs_l(iops1, eops1), rs_r(iops2, eops2);
	rs_l.readfile_gcps(path_gcps);
	rs_l.readfile_photos(path_left);
	rs_l.cal_resection("left");
	//rs_l.outer_precision_check();

	cout << endl << "！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！" << endl;
	cout << "Right Image processing..." << endl;
	rs_r.readfile_gcps(path_gcps);
	rs_r.readfile_photos(path_right);
	rs_r.cal_resection("right");
	//rs_r.outer_precision_check();

	//calculate rotation matrix
	Mat R1 = rs_l.Rotation(rs_l.omega, rs_l.phi, rs_l.kappa),
		R2 = rs_r.Rotation(rs_r.omega, rs_r.phi, rs_r.kappa);

	//calculate photogrammetric baseline components
	double Bx = rs_r.Xs - rs_l.Xs,
		By = rs_r.Ys - rs_l.Ys,
		Bz = rs_r.Zs - rs_l.Zs;

	double N1, N2, Xtp, Ytp, Ztp;
	ofstream f_output("result/Intersection.txt");
	rs_l.readfile_unknown(path_unknown);
	f_output << rs_l.id_unknown.size() << endl;

	//point projection
	Mat ll = Mat::zeros(4, 3, CV_64F), LL = Mat::zeros(4, 1, CV_64F);
	//(X1,Y1,Z) = R *��x1,y1,-f
	for (int i = 0; i < rs_l.id_unknown.size(); i++) {
		Mat tmp1 = Mat::zeros(3, 1, CV_64F),
			tmp2 = Mat::zeros(3, 1, CV_64F),
			p1 = Mat::zeros(3, 1, CV_64F),
			p2 = Mat::zeros(3, 1, CV_64F);
		tmp1.at<double>(0, 0) = rs_l.left_unknown[i].x;
		tmp1.at<double>(1, 0) = rs_l.left_unknown[i].y;
		tmp1.at<double>(2, 0) = -rs_l.f;
		tmp2.at<double>(0, 0) = rs_l.right_unknown[i].x;
		tmp2.at<double>(1, 0) = rs_l.right_unknown[i].y;
		tmp2.at<double>(2, 0) = -rs_r.f;
		//auxiliary coordinates in image space
		p1 = R1 * tmp1;
		p2 = R2 * tmp2;
		//calculate projection coefficient
		N1 = (Bx * p2.at<double>(2, 0) - Bz * p2.at<double>(0, 0)) / (p1.at<double>(0, 0) * p2.at<double>(2, 0) - p2.at<double>(0, 0) * p1.at<double>(2, 0));
		N2 = (Bx * p1.at<double>(2, 0) - Bz * p1.at<double>(0, 0)) / (p1.at<double>(0, 0) * p2.at<double>(2, 0) - p2.at<double>(0, 0) * p1.at<double>(2, 0));
		//calculate model point coordinates
		Xtp = rs_l.Xs + N1 * p1.at<double>(0, 0);
		Ytp = rs_l.Ys + N1 * p1.at<double>(1, 0);
		Ztp = rs_l.Zs + N1 * p1.at<double>(2, 0);

		PointGCP pt;
		pt.X = Xtp; pt.Y = Ytp; pt.Z = Ztp;
		rs_l.gcps_unknown.push_back(pt);
	}

	//rigorous collinearity model
	cout << endl << "！！！！！！！！！！！Ground Coordinates of Unknown Points ( rigorous collinearity model )！！！！！！！！！！！" << endl << endl;
	for (int i = 0; i < rs_l.id_unknown.size(); i++) {

		int flag = 0;

		double x1 = rs_l.left_unknown[i].x, y1 = rs_l.left_unknown[i].y, x2 = rs_l.right_unknown[i].x, y2 = rs_l.right_unknown[i].y;

		double rr = (x1 - rs_l.x0) * (x1 - rs_l.x0) + (y1 - rs_l.y0) * (y1 - rs_l.y0);
		x1 += ((x1 - rs_l.x0) * (rs_l.k1 * rr + rs_l.k2 * rr * rr) + rs_l.p1 * (rr + 2 * (x1 - rs_l.x0) * (x1 - rs_l.x0) + rs_l.p2 * 2 * (x1 - rs_l.x0) * (y1 - rs_l.y0)));
		y1 += ((y1 - rs_l.y0) * (rs_l.k1 * rr + rs_l.k2 * rr * rr) + rs_l.p1 * (rr + 2 * (y1 - rs_l.y0) * (y1 - rs_l.y0) + rs_l.p2 * 2 * (x1 - rs_l.x0) * (y1 - rs_l.y0)));

		rr = (x2 - rs_r.x0) * (x2 - rs_r.x0) + (y2 - rs_r.y0) * (y2 - rs_r.y0);
		x2 += ((x2 - rs_r.x0) * (rs_r.k1 * rr + rs_r.k2 * rr * rr) + rs_r.p1 * (rr + 2 * (x2 - rs_r.x0) * (x2 - rs_r.x0) + rs_l.p2 * 2 * (x2 - rs_r.x0) * (y2 - rs_r.y0)));
		y2 += ((y2 - rs_r.y0) * (rs_r.k1 * rr + rs_r.k2 * rr * rr) + rs_r.p1 * (rr + 2 * (y2 - rs_r.y0) * (y2 - rs_r.y0) + rs_l.p2 * 2 * (x2 - rs_r.x0) * (y2 - rs_r.y0)));

		ll.at<double>(0, 0) = rs_l.f * R1.at <double>(0, 0) + (x1 - rs_l.x0) * R1.at <double>(0, 2);
		ll.at<double>(0, 1) = rs_l.f * R1.at <double>(1, 0) + (x1 - rs_l.x0) * R1.at <double>(1, 2);
		ll.at<double>(0, 2) = rs_l.f * R1.at <double>(2, 0) + (x1 - rs_l.x0) * R1.at <double>(2, 2);
		ll.at<double>(1, 0) = rs_l.f * R1.at <double>(0, 1) + (y1 - rs_l.y0) * R1.at <double>(0, 2);
		ll.at<double>(1, 1) = rs_l.f * R1.at <double>(1, 1) + (y1 - rs_l.y0) * R1.at <double>(1, 2);
		ll.at<double>(1, 2) = rs_l.f * R1.at <double>(2, 1) + (y1 - rs_l.y0) * R1.at <double>(2, 2);

		LL.at<double>(0, 0) = (rs_l.f * R1.at <double>(0, 0) * rs_l.Xs + rs_l.f * R1.at <double>(1, 0) * rs_l.Ys + rs_l.f * R1.at <double>(2, 0) * rs_l.Zs
			+ (x1 - rs_l.x0) * R1.at <double>(0, 2) * rs_l.Xs
			+ (x1 - rs_l.x0) * R1.at <double>(1, 2) * rs_l.Ys
			+ (x1 - rs_l.x0) * R1.at <double>(2, 2) * rs_l.Zs);
		LL.at<double>(1, 0) = (rs_l.f * R1.at <double>(0, 1) * rs_l.Xs + rs_l.f * R1.at <double>(1, 1) * rs_l.Ys + rs_l.f * R1.at <double>(2, 1) * rs_l.Zs
			+ (y1 - rs_l.y0) * R1.at <double>(0, 2) * rs_l.Xs
			+ (y1 - rs_l.y0) * R1.at <double>(1, 2) * rs_l.Ys
			+ (y1 - rs_l.y0) * R1.at <double>(2, 2) * rs_l.Zs);

		ll.at<double>(2, 0) = rs_r.f * R2.at <double>(0, 0) + (x2 - rs_r.x0) * R2.at <double>(0, 2);
		ll.at<double>(2, 1) = rs_r.f * R2.at <double>(1, 0) + (x2 - rs_r.x0) * R2.at <double>(1, 2);
		ll.at<double>(2, 2) = rs_r.f * R2.at <double>(2, 0) + (x2 - rs_r.x0) * R2.at <double>(2, 2);
		ll.at<double>(3, 0) = rs_r.f * R2.at <double>(0, 1) + (y2 - rs_r.y0) * R2.at <double>(0, 2);
		ll.at<double>(3, 1) = rs_r.f * R2.at <double>(1, 1) + (y2 - rs_r.y0) * R2.at <double>(1, 2);
		ll.at<double>(3, 2) = rs_r.f * R2.at <double>(2, 1) + (y2 - rs_r.y0) * R2.at <double>(2, 2);

		LL.at<double>(2, 0) = (rs_r.f * R2.at <double>(0, 0) * rs_r.Xs + rs_r.f * R2.at <double>(1, 0) * rs_r.Ys + rs_r.f * R2.at <double>(2, 0) * rs_r.Zs
			+ (x2 - rs_r.x0) * R2.at <double>(0, 2) * rs_r.Xs
			+ (x2 - rs_r.x0) * R2.at <double>(1, 2) * rs_r.Ys
			+ (x2 - rs_r.x0) * R2.at <double>(2, 2) * rs_r.Zs);
		LL.at<double>(3, 0) = (rs_r.f * R2.at <double>(0, 1) * rs_l.Xs + rs_r.f * R2.at <double>(1, 1) * rs_r.Ys + rs_l.f * R2.at <double>(2, 1) * rs_r.Zs
			+ (y2 - rs_r.y0) * R2.at <double>(0, 2) * rs_r.Xs
			+ (y2 - rs_r.y0) * R2.at <double>(1, 2) * rs_r.Ys
			+ (y2 - rs_r.y0) * R2.at <double>(2, 2) * rs_r.Zs);

		Mat XYZ = (ll.t() * ll).inv() * ll.t() * LL;

		rs_l.gcps_unknown[i].X = XYZ.at<double>(0, 0);
		rs_l.gcps_unknown[i].Y = XYZ.at<double>(1, 0);
		rs_l.gcps_unknown[i].Z = XYZ.at<double>(2, 0);

		cout << rs_l.id_unknown[i] << " " << -rs_l.gcps_unknown[i].Z << "  " << rs_l.gcps_unknown[i].X << "  " << rs_l.gcps_unknown[i].Y << endl;

		f_output << rs_l.id_unknown[i] << "  " << -rs_l.gcps_unknown[i].Z << "  " << rs_l.gcps_unknown[i].X << "  " << rs_l.gcps_unknown[i].Y << endl;
	}

	f_output.close();

	cout << endl << "Success!" << endl;

	double VX, VY, VZ, V = 0;
	int flag = 0;
	cout << endl;
	ofstream fout("result/Intersection_PointCheck(mm).txt");
	cout << endl << "！！！！！！！！！！！Accuracy Assessment for Intersection！！！！！！！！！！！" << endl << endl;

	for (int i = 0; i < rs_l.id_unknown.size(); i++) {
		vector <int>::iterator iElement = find(rs_l.id_gcps.begin(), rs_l.id_gcps.end(), rs_l.id_unknown[i]);
		if (iElement != rs_l.id_gcps.end())
		{
			int nPosition = distance(rs_l.id_gcps.begin(), iElement);

			PointGCP tmp = rs_l.gcps[nPosition];

			VX = tmp.X - rs_l.gcps_unknown[i].X; VY = tmp.Y - rs_l.gcps_unknown[i].Y; VZ = tmp.Z - rs_l.gcps_unknown[i].Z;

			cout << "Point " << rs_l.id_unknown[i] << " " << VX << " " << VY << " " << VZ << endl;
			fout << rs_l.id_unknown[i] << " " << VX << " " << VY << " " << VZ << endl;

			V += sqrt(VX * VX + VY * VY + VZ * VZ);
			flag++;

		}
		else
			continue;
	}
	cout << endl << "Known Point Num:" << flag << endl;
	cout << "Point Error Means:" << V / flag << "\t(mm)" << endl;
	fout.close();

	cout << endl << "！！！！！！！！！！！！！ Finish! ！！！！！！！！！！！！！" << endl;

}

//DLT calculate interior & exterior orientation parameters and coordinates of unknown points
void DLT() {
	string path_gcps = "data/GCP.txt";
	string path_left = "data/left.txt";
	string path_right = "data/right.txt";
	string path_unkown = "data/pair_unknown.txt";

	CDLT dlt(path_left, path_right, path_gcps, path_unkown);
	cout << "！！！！！！！！！！！！！！！！！ Left Image ！！！！！！！！！！！！！！！！！" << endl;
	dlt.GetElement(dlt.id_photos_left, dlt.photos_left, "left");//L matrix, distortion coefficients, interior & exterior orientation parameters
	//dlt.outer_precision_check(dlt.id_photos_left, dlt.photos_left);
	cout << "！！！！！！！！！！！！！！！！！ Right Image ！！！！！！！！！！！！！！！！！" << endl;
	dlt.GetElement(dlt.id_photos_right, dlt.photos_right, "right");//L matrix, distortion coefficients, interior & exterior orientation parameters
	//dlt.outer_precision_check(dlt.id_photos_right, dlt.photos_right);

	cout << endl << "！！！！！！！！！！！！！ Get DLT Element Finish! ！！！！！！！！！！！！！" << endl;

	dlt.init_XYZ();
	dlt.cal_XYZ();
	dlt.XYZ_precision_check();

	cout << endl << "！！！！！！！！！！！！！ Finish! ！！！！！！！！！！！！！" << endl;

}

int main()
{
	Re_Intersection();

	DLT();

}