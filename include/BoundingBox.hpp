#ifndef BOUNDINGBOXES_HPP
#define BOUNDINGBOXES_HPP


class BoundingBox {
public:
    BoundingBox();

    unsigned short getHeight() {return height;}
    unsigned short getWidth() {return width;}
    unsigned short getTlCorner() {return tlCorner;}
    unsigned short getBrCorner() {return brCorner;}
    unsigned short getCenter() {return center;}
    unsigned short getNumber() {return number;}
    unsigned short getAngle() {return angle;}

private:
    unsigned short height;
    unsigned short width;
    unsigned short tlCorner;
    unsigned short brCorner;
    unsigned short center;
    unsigned short number;
    unsigned short angle;
    bool id;
};

#endif