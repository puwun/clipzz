"use client";

import Link from "next/link";
import Image from "next/image";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { signOut } from "next-auth/react";
import { useTheme } from "~/context/ThemeContext"; 
// import { Dices } from "lucide-react";
import { Palette } from 'lucide-react';



const NavHeader = ({ credits, email, image }: { credits: number; email: string; image: string }) => {
  const { setRandomTheme, currentThemeName } = useTheme();
  // console.log(credits, email,image)

  return (
    <header className="bg-background sticky top-0 z-10 flex justify-center border-b">
      <div className="container flex h-16 items-center justify-between px-4 py-2">
        <Link href="/dashboard" className="flex items-center">
          <Image
            src="/clipzz_favicon.jpeg"
            alt="Clipper Logo"
            className="h-14 w-14 ml-5 mr-4"
            width={56}
            height={56}
          />
          <span className="text-3xl font-bold text-primary">
            Clipzz
          </span>
        </Link>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Badge
              variant="secondary"
              className="h-8 px-3 py-1.5 text-xs font-medium"
            >
              {credits} credits
            </Badge>
            <Button
              variant="outline"
              size="sm"
              asChild
              className="h-8 text-xs font-medium"
            >
              <Link href="/dashboard/billing">Buy more</Link>
            </Button>
          </div>

          {/* 🎲 Theme Randomizer Button */}
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8"
            onClick={setRandomTheme}
            title={`Current theme: ${currentThemeName}`}
          >
            <Palette className="size-4" />
          </Button>

          {/* User Avatar Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className="relative h-8 w-8 rounded-full p-0"
              >
                <Avatar>
                  <AvatarFallback>{image ? <Image src={image} alt="" width={32} height={32} /> : email.charAt(0)}</AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>
                <p className="text-muted-foreground text-xs">{email}</p>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link href="/dashboard/billing">Billing</Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => signOut({ redirectTo: "/login" })}
                className="text-destructive cursor-pointer"
              >
                Sign out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
};

export default NavHeader;
